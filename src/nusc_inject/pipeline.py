"""Shared single-sample flow: Steps 1-6 + Step-5 spec for one sample.

Used by scripts/run_p0.py (one object) and scripts/run_batch.py (N objects,
many samples).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from . import anchor, annotation, camera_channel, composite, lidar_channel
from . import mesh_points, nusc_io, occluders as occ_mod
from . import validate, visualize

log = logging.getLogger("pipeline")


def _sample_objects(objects, cfg, cache):
    """Per unique model: surface-sample once and cache (pts, normals, bbox)."""
    for o in objects:
        path = o["model_cfg"]["path"]
        if path not in cache:
            pts, normals, bmin, bsize = mesh_points.sample_mesh(
                path, int(cfg["mesh_points"]["n_dense"]))
            cache[path] = {"pts": pts, "normals": normals,
                           "bbox_min": bmin, "bbox_size": bsize}
            log.info("sampled %s: raw bbox size=%s", Path(path).name,
                     np.round(bsize, 2))
    return cache


def run_sample(cfg, meta, sample_index, objects, out_dir: Path,
               mesh_cache=None, track_keys=None, drop_invisible=False) -> dict:
    """objects: [{"model_cfg": {...}, "placement_cfg": {...}}, ...]

    track_keys: optional list aligned with objects (cross-frame track id).
    drop_invisible: instead of aborting when an object has 0 lidar points or
        no camera sees the set, drop the failed objects (re-inject) and return
        a status so the caller can terminate those tracks.

    Returns {"status": "ok"|"no_visible"|"no_camera"|"validation_failed", ...}
    plus, on ok: {"sample", "gts", "lidar_result", "cam_result", "per_cam",
                  "report", "spec", "out_dir", "dropped"}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    mesh_cache = _sample_objects(objects, cfg, {} if mesh_cache is None else mesh_cache)
    track_keys = list(track_keys) if track_keys else [None] * len(objects)
    dropped = []  # original object indices dropped as invisible

    sample = meta.sample(int(sample_index))
    lidar_sd = meta.sd_for_sample(sample["token"], cfg["dataset"]["lidar_channel"])
    if not lidar_sd["filename"].startswith("samples/"):
        raise SystemExit(f"FATAL: not a keyframe lidar file: {lidar_sd['filename']}")
    scene_pts = nusc_io.load_lidar_bin(meta.sensor_path(lidar_sd))
    log.info("sample %d token=%s scene=%s cloud=%d pts", sample_index,
             sample["token"][:8], meta.scene[sample["scene_token"]]["name"],
             scene_pts.shape[0])

    # ---- Step 1: anchor (per object) ----
    cfg["dataset"]["sample_index"] = int(sample_index)
    gts = []
    for o in objects:
        m = mesh_cache[o["model_cfg"]["path"]]
        gt = anchor.build_gt(cfg, meta, scene_pts, sample=sample,
                             model_cfg=o["model_cfg"],
                             placement_cfg=o["placement_cfg"],
                             raw_bbox=(m["bbox_min"], m["bbox_size"]))
        gts.append(gt)
    for i, gt in enumerate(gts):
        log.info("obj%d: pos=%s size=%s scale=%.5f ground_z=%.3f", i,
                 np.round(gt["translation"], 2), np.round(gt["size"], 2),
                 gt["scale"], gt["ground_z_lidar"])

    def _payload():
        return [
            {"pts_raw": mesh_cache[gt["model_path"]]["pts"],
             "normals_raw": mesh_cache[gt["model_path"]]["normals"],
             "gt": gt}
            for gt in gts
        ]

    # ---- scene occluders (dense bbox bodies for the occlusion z-buffers) ----
    T_l2g = np.array(gts[0]["T_lidar2global"], dtype=float).reshape(4, 4)
    occ_boxes = occ_mod.build_scene_occluder_boxes(meta, sample["token"], cfg, T_l2g)
    log.info("scene occluder boxes: %d (sample_annotation)", len(occ_boxes))

    # ---- Step 2: LiDAR injection (joint) ----
    lr = lidar_channel.inject(scene_pts, _payload(), cfg, occluder_boxes=occ_boxes)
    if drop_invisible and any(n == 0 for n in lr["num_obj_kept_per_object"]):
        failed = [i for i, n in enumerate(lr["num_obj_kept_per_object"]) if n == 0]
        keep = [i for i in range(len(objects)) if i not in failed]
        log.warning("invisible objects %s dropped (0 lidar pts); survivors: %d",
                    failed, len(keep))
        dropped.extend(failed)
        if not keep:
            return {"status": "no_visible", "dropped": dropped,
                    "sample": sample, "out_dir": out_dir}
        objects = [objects[i] for i in keep]
        gts = [gts[i] for i in keep]
        track_keys = [track_keys[i] for i in keep]
        lr = lidar_channel.inject(scene_pts, _payload(), cfg, occluder_boxes=occ_boxes)
    if any(n == 0 for n in lr["num_obj_kept_per_object"]):
        if drop_invisible:
            return {"status": "no_visible", "dropped": dropped,
                    "sample": sample, "out_dir": out_dir}
        raise SystemExit(f"FATAL: an object survived with 0 points: "
                         f"{lr['num_obj_kept_per_object']}")

    fmt = cfg["lidar_channel"].get("out_format", "n5")
    bin_name = "injected.pcd.bin" if fmt == "n5" else "injected.bin"
    nusc_io.save_lidar_bin(out_dir / "lidar" / bin_name, lr["injected"])
    np.save(out_dir / "lidar" / "obj_mask.npy", lr["obj_mask"])
    np.save(out_dir / "lidar" / "obj_points_lidar.npy", lr["obj_points_lidar"])
    np.save(out_dir / "lidar" / "obj_ids.npy", lr["obj_ids"])
    log.info("LiDAR: scene %d + obj %d (per-object %s)",
             lr["num_scene_kept"], lr["num_obj_kept"],
             lr["num_obj_kept_per_object"])

    with open(out_dir / "gt.json", "w") as f:
        json.dump(gts if len(gts) > 1 else gts[0], f, indent=2)

    # ---- Step 3: camera render (one Blender job) ----
    try:
        cam_result = camera_channel.run(cfg, meta, gts, out_dir)
    except RuntimeError as e:
        if drop_invisible:
            log.warning("no camera sees objects: %s", e)
            return {"status": "no_camera", "dropped": dropped,
                    "sample": sample, "out_dir": out_dir}
        raise
    log.info("rendered cams: %s (max err %.4f px)",
             [c["name"] for c in cam_result["cams"]], cam_result["max_pixel_err"])

    # trimesh/Blender model-frame drift guard (per rendered object)
    for i, ib in enumerate(cam_result.get("imported_bboxes_raw", [])):
        gt = gts[i]
        bmin_t = np.array(gt["raw_bbox_min"])
        bsize_t = np.array(gt["raw_bbox_size"])
        bmin_b = np.array(ib["min"])
        bsize_b = np.array(ib["max"]) - np.array(ib["min"])
        drift = float(max(np.abs(bmin_t - bmin_b).max(),
                          np.abs(bsize_t - bsize_b).max()))
        if drift > 1e-3:
            raise SystemExit(
                f"FATAL: trimesh/Blender model-frame mismatch obj{i} "
                f"({ib['model_path']}): drift {drift:.4f} m")

    # ---- Step 4: compositing ----
    comp_dir = out_dir / "composite"
    depth_dir = out_dir / "depth"
    comp_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    per_cam = {}
    for c in cam_result["cams"]:
        name = c["name"]
        cam_spec = cam_result["cam_specs"][name]
        sd = meta.sd_for_sample(sample["token"], name)
        bg = cv2.cvtColor(cv2.imread(str(meta.sensor_path(sd))), cv2.COLOR_BGR2RGB)
        fg_path = out_dir / "blender" / f"{name}.png"
        fg = cv2.imread(str(fg_path), cv2.IMREAD_UNCHANGED)
        if fg is None:
            log.warning("no foreground for %s, skip", name)
            continue
        fg = cv2.cvtColor(fg, cv2.COLOR_BGRA2RGBA) if fg.shape[2] == 4 else fg

        out_rgb, visible, alpha, obj_depth, scene_depth, info = composite.composite_onto(
            name, fg, bg, _payload(), cam_spec, cfg, scene_pts,
            T_lidar2global=T_l2g, occluder_boxes=occ_boxes
        )
        # frustum margin may select cameras where the object barely misses the
        # frame (empty/tiny silhouette) — nothing to validate or replace
        min_alpha = int(cfg.get("validation", {}).get("min_alpha_px", 500))
        if info["n_alpha_px"] < min_alpha:
            log.info("  %s: alpha px=%d < %d, skip camera", name,
                     info["n_alpha_px"], min_alpha)
            continue
        cv2.imwrite(str(comp_dir / f"{name}.jpg"),
                    cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
        np.save(depth_dir / f"{name}_obj_depth.npy", obj_depth)
        np.save(depth_dir / f"{name}_scene_depth.npy", scene_depth)
        per_cam[name] = {"cam_spec": cam_spec, "alpha": alpha,
                         "visible": visible, "n_visible_px": info["n_visible_px"]}
        log.info("  %s: visible=%d alpha=%d occluded=%d", name,
                 info["n_visible_px"], info["n_alpha_px"], info["n_occluded_px"])
    if not per_cam:
        raise SystemExit("FATAL: no camera composited")

    # ---- Step 6: validation ----
    report = validate.run(cfg, gts, lr, cam_result, per_cam)
    with open(out_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("validation all_pass=%s", report["all_pass"])
    if not report["all_pass"] and drop_invisible:
        return {"status": "validation_failed", "report": report,
                "dropped": dropped, "sample": sample, "out_dir": out_dir}

    # ---- visualization ----
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    for name, pc in per_cam.items():
        comp_img = cv2.cvtColor(cv2.imread(str(comp_dir / f"{name}.jpg")),
                                cv2.COLOR_BGR2RGB)
        visualize.overlay_cam(pc["cam_spec"], comp_img, pc["alpha"],
                              lr["obj_points_lidar"], T_l2g,
                              viz_dir / f"{name}_overlay.jpg")
    visualize.lidar_bev(scene_pts[:, :3], lr["obj_points_lidar"][:, :3],
                        viz_dir / "lidar_bev.png")

    # ---- Step-5 spec (for annotation export) ----
    files = {
        "lidar_filename": lidar_sd["filename"],
        "lidar_output_path": str(out_dir / "lidar" / bin_name),
        "cam_filenames": {n: meta.sd_for_sample(sample["token"], n)["filename"]
                          for n in per_cam},
        "cam_output_paths": {n: str(comp_dir / f"{n}.jpg") for n in per_cam},
    }
    spec = annotation.build_sample_spec(sample, gts, lr, files,
                                        track_keys=track_keys)
    with open(out_dir / "spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    return {"status": "ok", "sample": sample, "gts": gts, "lidar_result": lr,
            "cam_result": cam_result, "per_cam": per_cam, "report": report,
            "spec": spec, "out_dir": out_dir, "dropped": dropped}
