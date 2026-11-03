#!/usr/bin/env python
"""Batch injection: per-sample objects (legacy) or cross-frame TRACKS.

Tracking mode (objects.tracking.enabled): each track is one world-static
object placed at a start frame, frozen T_model2global across consecutive
samples of the same scene, annotated with prev/next chains (one instance,
nbr_annotations = frames survived). Tracks terminate early when the object
becomes invisible (0 lidar pts / no camera) or a frame fails validation.

Usage:
    python scripts/run_batch.py --config configs/batch.yaml
    python scripts/run_batch.py --config configs/batch.yaml --samples 0 1 2   # legacy
    python scripts/run_batch.py --config configs/batch.yaml --export-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from nusc_inject import annotation, nusc_io, pipeline
from nusc_inject.config import load_config

ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("run_batch")


# ---------------------------------------------------------------- planning

def _plan_one(cfg, rng, overrides=None):
    """Random model + ego-frame placement for one object/track start."""
    oc = dict(cfg["objects"])
    if overrides:
        oc.update(overrides)
    fwd_lo, fwd_hi = [float(v) for v in oc.get("forward_range_m", (8.0, 30.0))]
    lat_lo, lat_hi = [float(v) for v in oc.get("lateral_range_m", (-4.0, 4.0))]
    yaw_lo, yaw_hi = [float(v) for v in oc.get("yaw_in_ego_deg_range", (0.0, 360.0))]
    ground = oc.get("ground", {"search_radius_m": 4.0, "z_percentile": 15.0})

    fwd = float(rng.uniform(fwd_lo, fwd_hi))
    lat = float(rng.uniform(lat_lo, lat_hi))
    yaw = float(rng.uniform(yaw_lo, yaw_hi))

    pool = cfg["models"]
    model = dict(pool[int(rng.integers(0, len(pool)))])
    dim = model.get("target_max_dim_m", 3.0)
    if isinstance(dim, (list, tuple)):
        model["target_max_dim_m"] = float(rng.uniform(float(dim[0]), float(dim[1])))

    return {
        "model_cfg": model,
        "placement_cfg": {
            "strategy": "random_forward",
            "offset_in_ego": [fwd, lat, 0.0],
            "yaw_in_ego_deg": yaw,
            "ground": ground,
        },
    }


def plan_objects(cfg, sample_index: int, rng: np.random.Generator):
    """Legacy (non-tracking): N independent objects for ONE sample."""
    n = int(cfg["objects"].get("count", 1))
    spacing = float(cfg["objects"].get("min_spacing_m", 6.0))
    objects, xy = [], []
    for _ in range(n):
        for _attempt in range(200):
            o = _plan_one(cfg, rng)
            fx, fy = o["placement_cfg"]["offset_in_ego"][:2]
            if all(np.hypot(fx - px, fy - py) >= spacing for px, py in xy):
                break
        xy.append((fx, fy))
        objects.append(o)
    return objects


def plan_tracks(cfg, scene_name, n_frames, rng):
    """Cross-frame tracks for ONE scene (static in the world frame)."""
    oc = cfg["objects"]
    tc = oc.get("tracking", {})
    n_tracks = int(tc.get("count", oc.get("count", 2)))
    lo, hi = [int(v) for v in tc.get("length", (3, 8))]
    overrides = {}
    if "forward_range_m" in tc:
        overrides["forward_range_m"] = tc["forward_range_m"]
    if "lateral_range_m" in tc:
        overrides["lateral_range_m"] = tc["lateral_range_m"]

    tracks = []
    for i in range(n_tracks):
        length = int(rng.integers(lo, hi + 1))
        length = max(1, min(length, n_frames))
        max_start = n_frames - length
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        one = _plan_one(cfg, rng, overrides)
        tracks.append({
            "key": f"{scene_name}_t{i:02d}",
            "start": start,
            "length": length,
            "model_cfg": one["model_cfg"],
            "placement_cfg": one["placement_cfg"],
            "T": None,            # resolved at first annotated frame
            "terminated": False,
            "end_reason": None,
        })
    return tracks


# ---------------------------------------------------------------- runners

def run_tracking(cfg, meta, batch_dir: Path):
    seed = int(cfg["objects"].get("seed", 42))
    scene_filter = cfg["dataset"].get("scenes")
    scenes = [s for s in meta.scene.values()
              if scene_filter is None or s["name"] in scene_filter]
    scenes.sort(key=lambda s: meta.sample_idx_by_token[s["first_sample_token"]])
    if not scenes:
        sys.exit(f"no scenes matched filter {scene_filter}")

    specs, track_stats = [], []
    total_ok = 0
    for si, scene in enumerate(scenes):
        samples = meta.scene_samples(scene)
        rng = np.random.default_rng([seed, si])
        tracks = plan_tracks(cfg, scene["name"], len(samples), rng)
        log.info("=== scene %s (%d frames, %d tracks) ===",
                 scene["name"], len(samples), len(tracks))

        mesh_cache = {}
        for fi, sample in enumerate(samples):
            active = [t for t in tracks
                      if not t["terminated"]
                      and t["start"] <= fi < t["start"] + t["length"]]
            if not active:
                continue
            objects, tkeys = [], []
            for t in active:
                if t["T"] is None:
                    objects.append({"model_cfg": t["model_cfg"],
                                    "placement_cfg": t["placement_cfg"]})
                else:
                    objects.append({
                        "model_cfg": t["model_cfg"],
                        "placement_cfg": {
                            "from_global": t["T"],
                            "ground": t["placement_cfg"].get("ground", {}),
                        },
                    })
                tkeys.append(t["key"])

            out_dir = batch_dir / scene["name"] / f"f{fi:03d}"
            idx = meta.sample_idx_by_token[sample["token"]]
            res = pipeline.run_sample(cfg, meta, idx, objects, out_dir,
                                      mesh_cache=mesh_cache,
                                      track_keys=tkeys, drop_invisible=True)
            status = res.get("status", "ok")
            dropped_keys = {tkeys[i] for i in res.get("dropped", [])}

            if status == "no_visible":
                for t in active:
                    t["terminated"] = True
                    t["end_reason"] = "invisible"
                log.info("  frame %d: all tracks invisible -> close", fi)
                continue
            if status in ("no_camera", "validation_failed"):
                for t in active:
                    t["terminated"] = True
                    t["end_reason"] = status
                log.info("  frame %d: %s -> close %d tracks",
                         fi, status, len(active))
                continue

            # ok — drop any track that lost visibility this frame
            for t in active:
                if t["key"] in dropped_keys:
                    t["terminated"] = True
                    t["end_reason"] = "invisible"

            # resolve frozen world pose for newly annotated tracks
            ann_keys = [o["track_key"] for o in res["spec"]["objects"]]
            for t in active:
                if t["T"] is None and t["key"] in ann_keys:
                    gi = ann_keys.index(t["key"])
                    t["T"] = res["gts"][gi]["T_model2global"]
                if t["key"] not in ann_keys and t["T"] is None:
                    # never got annotated (dropped at first frame)
                    t["terminated"] = True
                    t["end_reason"] = "invisible"

            specs.append(res["spec"])
            total_ok += 1
            log.info("  frame %d annotated: %d objects (%s)",
                     fi, len(ann_keys), ann_keys)

        for t in tracks:
            if not t["terminated"]:
                t["end_reason"] = "window_end"
            track_stats.append({
                "key": t["key"], "start": t["start"], "planned_length": t["length"],
                "end_reason": t["end_reason"], "T_resolved": t["T"] is not None,
            })

    lengths = Counter(o["track_key"] for s in specs for o in s["objects"])
    summary = {
        "scenes": [s["name"] for s in scenes],
        "frames_annotated": total_ok,
        "n_tracks_planned": len(track_stats),
        "n_tracks_realized": sum(1 for k in lengths),
        "track_length_hist": dict(Counter(lengths.values())),
        "end_reasons": dict(Counter(t["end_reason"] for t in track_stats)),
    }
    return specs, summary


def run_legacy_samples(cfg, meta, batch_dir: Path):
    """N independent objects per sample.

    ``drop_invisible=True`` matches the tracking path: an object the scene
    fully occludes is dropped instead of aborting the sample, so a correctly
    hidden object no longer fails a whole frame. The sample is skipped only
    when nothing survives, no camera sees the set, or validation fails.
    """
    seed = int(cfg["objects"].get("seed", 42))
    sample_indices = [int(i) for i in cfg["dataset"]["sample_indices"]]
    specs, failed, dropped_log = [], [], []
    for idx in sample_indices:
        rng = np.random.default_rng([seed, idx])
        objects = plan_objects(cfg, idx, rng)
        out_dir = batch_dir / f"sample_{idx:04d}"
        log.info("=== sample %d: %d objects ===", idx, len(objects))
        try:
            res = pipeline.run_sample(cfg, meta, idx, objects, out_dir,
                                      drop_invisible=True)
        except SystemExit as e:
            log.error("sample %d failed: %s", idx, e)
            failed.append({"sample_index": idx, "error": str(e)})
            continue
        status = res.get("status", "ok")
        if status != "ok":
            log.warning("sample %d: %s", idx, status)
            failed.append({"sample_index": idx, "error": status})
            continue
        dropped = list(res.get("dropped", []))
        if dropped:
            log.info("sample %d: dropped %d occluded object(s) %s",
                     idx, len(dropped), dropped)
            dropped_log.append({"sample_index": idx, "dropped": dropped})
        specs.append(res["spec"])
    summary = {"sample_indices": sample_indices, "n_ok": len(specs),
               "failed": failed,
               "n_objects_dropped": sum(len(d["dropped"]) for d in dropped_log),
               "dropped": dropped_log}
    return specs, summary


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "batch.yaml"))
    ap.add_argument("--samples", type=int, nargs="*", default=None,
                    help="override sample_indices (forces legacy mode)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, ROOT)
    if args.run_id:
        cfg["run_id"] = args.run_id
    if args.samples is not None:
        cfg["dataset"]["sample_indices"] = args.samples
        cfg.setdefault("objects", {}).setdefault("tracking", {})["enabled"] = False

    batch_dir = ROOT / cfg.get("output_dir", "data/outputs/p0") / cfg["run_id"]
    batch_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(batch_dir / "batch.log", mode="w"),
                  logging.StreamHandler()],
    )

    manifest_path = batch_dir / "specs.json"
    specs = None
    summary = None
    if args.export_only:
        if not manifest_path.exists():
            sys.exit(f"no {manifest_path}; run the batch first")
        with open(manifest_path) as f:
            man = json.load(f)
        specs, summary = man["specs"], man.get("summary")
        log.info("export-only: %d sample specs", len(specs))
    else:
        meta = nusc_io.NuScenesMini(cfg["dataset"]["root"], cfg["dataset"]["version"])
        tracking = cfg.get("objects", {}).get("tracking", {}).get("enabled", False)
        if tracking:
            specs, summary = run_tracking(cfg, meta, batch_dir)
        else:
            specs, summary = run_legacy_samples(cfg, meta, batch_dir)
        with open(manifest_path, "w") as f:
            json.dump({"run_id": cfg["run_id"], "summary": summary,
                       "specs": specs}, f, indent=2)
        log.info("batch done: %d annotated frames", len(specs))
        log.info("summary: %s", json.dumps(summary, ensure_ascii=False))
        if not specs:
            sys.exit("no annotated frames; nothing to export")

    if args.skip_export:
        print("batch done (export skipped):", batch_dir)
        return 0

    # ---- Step 5: export + verify ----
    export_dir = ROOT / cfg.get("export_dir", "data/exports/nuscenes-od-p0")
    n_expected = sum(len(s["objects"]) for s in specs)
    out = annotation.export_dataset(cfg, specs, export_dir)
    log.info("exported: %s", out)

    report = annotation.verify_export(
        export_dir, cfg["dataset"]["version"], n_expected,
        category_names=out.get("category_names"),
        instance_tokens=out.get("instance_tokens"))
    log.info("verify: %s", report)
    with open(export_dir / "export_report.json", "w") as f:
        json.dump({"summary": summary, "export": out, "verify": report},
                  f, indent=2)

    ok = report["ok"] and out["n_new_annotations"] == n_expected
    print("\n=== batch export:", export_dir)
    print("new annotations:", out["n_new_annotations"],
          " tracks:", report.get("n_tracks", 0),
          " verified:", report["ok"])
    if report.get("problems"):
        print("problems:", report["problems"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
