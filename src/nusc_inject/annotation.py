"""Step 5 — nuScenes-standard annotation records + dataset export.

Pipeline side writes per-sample SPECs (gt summary + injected file names).
``export_dataset`` then builds the standard tables (category / attribute /
instance / sample_annotation), materializes an export dataroot (metadata +
samples tree with injected files; sweeps/maps symlinked), ready for
``nuscenes-devkit``.

Tokens are deterministic (sha256 over a namespace + identities) so repeated
exports of the same specs produce identical token graphs.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

TABLES = [
    "attribute", "calibrated_sensor", "category", "ego_pose", "instance", "log",
    "map", "sample_annotation", "sample_data", "sample", "scene", "sensor",
    "visibility",
]


def det_token(*parts) -> str:
    h = hashlib.sha256(("injectnusc|" + "|".join(str(p) for p in parts)).encode())
    return h.hexdigest()[:32]


def build_sample_spec(sample, gts, lidar_result, files: dict,
                      track_keys=None) -> dict:
    """Per-sample record consumed by export_dataset (Step-5 pipeline output).

    files: {
      "lidar_filename":  original rel path  (e.g. samples/LIDAR_TOP/x.pcd.bin),
      "lidar_output_path": local injected bin,
      "cam_filenames":   {cam: original rel path},
      "cam_output_paths":{cam: local composited jpg},
    }
    track_keys: optional list aligned with gts — stable key for cross-frame
    tracks (objects sharing a key form one instance with prev/next links).
    """
    return {
        "sample_token": sample["token"],
        "scene_token": sample["scene_token"],
        "timestamp": sample["timestamp"],
        "lidar_filename": files.get("lidar_filename"),
        "lidar_output_path": files.get("lidar_output_path"),
        "cam_filenames": dict(files.get("cam_filenames", {})),
        "cam_output_paths": dict(files.get("cam_output_paths", {})),
        "objects": [
            {
                "translation": gt["translation"],
                "size": gt["size"],
                "rotation": gt["rotation"],
                "category": gt["category"],
                "model_path": gt["model_path"],
                "num_lidar_pts": int(n),
                "ground_z_lidar": gt["ground_z_lidar"],
                "track_key": (track_keys[i] if track_keys else None),
            }
            for i, (gt, n) in enumerate(
                zip(gts, lidar_result["num_obj_kept_per_object"]))
        ],
    }


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _dump_json(path, data):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _ensure_category(categories, name, description):
    for c in categories:
        if c["name"] == name:
            return c["token"]
    tok = det_token("category", name)
    categories.append({"token": tok, "name": name, "description": description})
    return tok


def _ensure_attribute(attributes, name, description):
    for a in attributes:
        if a["name"] == name:
            return a["token"]
    tok = det_token("attribute", name)
    attributes.append({"token": tok, "name": name, "description": description})
    return tok


def export_dataset(cfg, specs, export_root) -> dict:
    """Build a nuScenes-format export with injected objects.

    cfg: full config (needs dataset.root / dataset.version and annotation.*).
    specs: list of build_sample_spec outputs.
    Returns {"export_root", "n_new_annotations", "category_token", ...}.
    """
    ann_cfg = cfg.get("annotation", {})
    default_category = ann_cfg.get("category_name", "movable_object.anomaly")
    attribute_name = ann_cfg.get("attribute_name", "vehicle.moving")
    visibility_token = str(ann_cfg.get("visibility_token", "4"))

    orig_root = Path(cfg["dataset"]["root"])
    version = cfg["dataset"]["version"]
    orig_meta = orig_root / version
    export_root = Path(export_root).resolve()
    if export_root == orig_root.resolve():
        raise ValueError("export_root must differ from the original dataset root")

    out_meta = export_root / version
    if export_root.exists():
        shutil.rmtree(export_root)
    out_meta.mkdir(parents=True)

    # --- 1) tables: copy originals, then append our rows ---
    for name in TABLES:
        src = orig_meta / f"{name}.json"
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy(src, out_meta / f"{name}.json")

    categories = _load_json(out_meta / "category.json")
    attributes = _load_json(out_meta / "attribute.json")
    instances = _load_json(out_meta / "instance.json")
    anns = _load_json(out_meta / "sample_annotation.json")
    visibilities = _load_json(out_meta / "visibility.json")
    if visibility_token not in {v["token"] for v in visibilities}:
        raise ValueError(f"visibility_token {visibility_token!r} not in visibility.json")

    # Each object may carry its own nuScenes category name (``model_cfg.category``);
    # objects without one fall back to ``annotation.category_name``. Previously a
    # single global category was used for every injected object, so the per-model
    # field was silently dropped.
    cat_tokens: dict = {}

    def _category_for(obj) -> str:
        return obj.get("category") or default_category

    def _category_token(obj) -> str:
        name = _category_for(obj)
        if name not in cat_tokens:
            cat_tokens[name] = _ensure_category(
                categories, name,
                "third-party object injected by nusc-inject")
        return cat_tokens[name]

    attribute_token = _ensure_attribute(
        attributes, attribute_name, "attribute of injected objects")

    n_new = 0
    # The exact rows we add. Verification selects by token, not by category:
    # an injected object may reuse an EXISTING nuScenes category (e.g.
    # human.pedestrian.adult), and selecting by category would then sweep in
    # every real annotation of that class.
    created_ann_tokens: list = []
    created_inst_tokens: list = []

    def _ann_token(spec, oi):
        return det_token("annotation", spec["sample_token"], oi)

    def _add_ann(spec, obj, inst_token, ann_token, prev, next_):
        anns.append({
            "token": ann_token,
            "sample_token": spec["sample_token"],
            "instance_token": inst_token,
            "visibility_token": visibility_token,
            "attribute_tokens": [attribute_token],
            "translation": [float(v) for v in obj["translation"]],
            "size": [float(v) for v in obj["size"]],
            "rotation": [float(v) for v in obj["rotation"]],
            "prev": prev,
            "next": next_,
            "num_lidar_pts": int(obj["num_lidar_pts"]),
            "num_radar_pts": 0,
        })

    # --- cross-frame tracks: one instance, prev/next chain over frames ---
    tracks = {}  # track_key -> [(spec_idx, obj_idx), ...] in temporal order
    for si, spec in enumerate(specs):
        for oi, obj in enumerate(spec["objects"]):
            key = obj.get("track_key")
            if key:
                tracks.setdefault(key, []).append((si, oi))

    seen_single = set()
    for key, members in tracks.items():
        names = {_category_for(specs[si]["objects"][oi]) for si, oi in members}
        if len(names) != 1:
            raise ValueError(
                f"track {key!r} spans multiple categories {sorted(names)}; "
                f"a nuScenes instance has exactly one category")
        inst_token = det_token("instance", key)
        ann_tokens = [_ann_token(specs[si], oi) for si, oi in members]
        instances.append({
            "token": inst_token,
            "category_token": _category_token(specs[members[0][0]]["objects"][members[0][1]]),
            "nbr_annotations": len(members),
            "first_annotation_token": ann_tokens[0],
            "last_annotation_token": ann_tokens[-1],
        })
        created_inst_tokens.append(inst_token)
        created_ann_tokens.extend(ann_tokens)
        for j, ((si, oi), ann_tok) in enumerate(zip(members, ann_tokens)):
            _add_ann(specs[si], specs[si]["objects"][oi], inst_token, ann_tok,
                     ann_tokens[j - 1] if j > 0 else "",
                     ann_tokens[j + 1] if j + 1 < len(ann_tokens) else "")
            n_new += 1
        seen_single.update(members)

    # --- single-frame objects (no track_key): own instance, empty links ---
    for si, spec in enumerate(specs):
        for oi, obj in enumerate(spec["objects"]):
            if (si, oi) in seen_single or obj.get("track_key"):
                continue
            inst_token = det_token("instance", spec["sample_token"], oi)
            ann_token = _ann_token(spec, oi)
            instances.append({
                "token": inst_token,
                "category_token": _category_token(obj),
                "nbr_annotations": 1,
                "first_annotation_token": ann_token,
                "last_annotation_token": ann_token,
            })
            created_inst_tokens.append(inst_token)
            created_ann_tokens.append(ann_token)
            _add_ann(spec, obj, inst_token, ann_token, "", "")
            n_new += 1

    _dump_json(out_meta / "category.json", categories)
    _dump_json(out_meta / "attribute.json", attributes)
    _dump_json(out_meta / "instance.json", instances)
    _dump_json(out_meta / "sample_annotation.json", anns)

    # --- 2) samples tree: per-file symlinks, injected files copied in ---
    replacements = {}
    for spec in specs:
        if spec["lidar_filename"]:
            if not spec["lidar_filename"].startswith("samples/"):
                raise ValueError(f"lidar_filename not under samples/: "
                                 f"{spec['lidar_filename']}")
            replacements[spec["lidar_filename"]] = spec.get("lidar_output_path")
        for cam, fname in spec.get("cam_filenames", {}).items():
            if not fname.startswith("samples/"):
                raise ValueError(f"cam filename not under samples/: {fname}")
            replacements[fname] = spec.get("cam_output_paths", {}).get(cam)

    n_files = 0
    for dirpath, _dirnames, filenames in os.walk(orig_root / "samples"):
        for fn in filenames:
            rel = str((Path(dirpath) / fn).relative_to(orig_root))
            dst = export_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            repl = replacements.get(rel)
            if repl and Path(repl).exists():
                shutil.copy(repl, dst)   # real injected file
            else:
                dst.symlink_to((orig_root / rel).resolve())
            n_files += 1

    # symlinks for untouched heavy dirs (devkit walks sweeps/, maps/)
    for d in ("sweeps", "maps"):
        src = orig_root / d
        if src.exists():
            (export_root / d).symlink_to(src.resolve(), target_is_directory=True)

    # Provenance travels with the dataset. export_root is rebuilt from scratch
    # on every export, so the manifest describing WHAT was injected has to live
    # inside it — otherwise a re-export silently destroys the only record of
    # the injected objects (positions, sizes, categories, track keys).
    _dump_json(export_root / "specs.json", {
        "run_id": cfg.get("run_id"),
        "n_samples": len(specs),
        "specs": specs,
    })

    return {
        "export_root": str(export_root),
        "version": version,
        "n_new_annotations": n_new,
        "n_samples_injected": len(specs),
        "category_names": sorted(cat_tokens),
        "category_tokens": dict(cat_tokens),
        "annotation_tokens": created_ann_tokens,
        "instance_tokens": created_inst_tokens,
        "attribute_token": attribute_token,
        "n_sample_files": n_files,
    }


def patch_devkit_colormap(nusc, category_name: str = "movable_object.anomaly",
                          rgb=(255, 0, 255)) -> None:
    """Register a custom category in a NuScenes instance's colormap.

    The stock devkit render_* helpers look up ``nusc.colormap[category.name]``
    (a per-instance dict built by get_colormap()) and raise KeyError for
    categories added by this project. Call after constructing NuScenes() and
    before nusc.render_sample() / render_pointcloud in downstream scripts.
    """
    nusc.colormap.setdefault(category_name, rgb)


def verify_export(export_root, version, n_expected, category_names=None,
                  instance_tokens=None) -> dict:
    """Load the export with nuscenes-devkit and check our rows are queryable.

    ``instance_tokens`` is the authoritative selector: the exact instances
    ``export_dataset`` added. Selecting by token rather than by category is
    required because an injected object may reuse an **existing** nuScenes
    category (e.g. ``human.pedestrian.adult``) — a category-based query would
    then sweep in every real annotation of that class and report thousands of
    unrelated rows.

    ``category_names`` only affects colormap registration for the devkit
    render helpers.
    """
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(export_root), verbose=False)
    names = set(category_names or [])
    for name in names:
        patch_devkit_colormap(nusc, name)

    if instance_tokens is not None:
        wanted = set(instance_tokens)
        new_inst = [i for i in nusc.instance if i["token"] in wanted]
        if len(new_inst) != len(wanted):
            missing = len(wanted) - len(new_inst)
            raise RuntimeError(
                f"{missing} injected instance(s) missing from the export")
    else:
        cat_token_by_name = {c["name"]: c["token"] for c in nusc.category}
        new_cat_tokens = ({cat_token_by_name[n] for n in names
                           if n in cat_token_by_name} if names else
                          {t for n, t in cat_token_by_name.items() if "anomaly" in n})
        new_inst = [i for i in nusc.instance if i["category_token"] in new_cat_tokens]

    new_inst_tokens = {i["token"] for i in new_inst}
    new_anns = [a for a in nusc.sample_annotation
                if a["instance_token"] in new_inst_tokens]
    problems = []
    if len(new_anns) != n_expected:
        problems.append(f"expected {n_expected} new anns, found {len(new_anns)}")
    for a in new_anns:
        if a["num_lidar_pts"] <= 0:
            problems.append(f"ann {a['token']} has 0 lidar pts")
        if a["visibility_token"] not in ("1", "2", "3", "4"):
            problems.append(f"bad visibility {a['visibility_token']}")

    # cross-frame track integrity: chain length, order, static pose
    ann_by_token = {a["token"]: a for a in nusc.sample_annotation}
    sample_by_token = {s["token"]: s for s in nusc.sample}
    n_tracks = 0
    for inst in new_inst:
        if inst["nbr_annotations"] <= 1:
            continue
        n_tracks += 1
        chain, tok = [], inst["first_annotation_token"]
        while tok:
            if tok in chain:
                problems.append(f"track {inst['token']}: cycle at {tok}")
                break
            chain.append(tok)
            nxt = ann_by_token[tok]["next"]
            if len(chain) > inst["nbr_annotations"] + 1:
                problems.append(f"track {inst['token']}: chain too long")
                break
            tok = nxt
        if len(chain) != inst["nbr_annotations"]:
            problems.append(
                f"track {inst['token']}: chain {len(chain)} != "
                f"nbr_annotations {inst['nbr_annotations']}")
        if chain and chain[-1] != inst["last_annotation_token"]:
            problems.append(f"track {inst['token']}: last token mismatch")
        # temporal order + static translation (tracks are world-static here)
        stamps = [sample_by_token[ann_by_token[t]["sample_token"]]["timestamp"]
                  for t in chain]
        if stamps != sorted(stamps) or len(set(stamps)) != len(stamps):
            problems.append(f"track {inst['token']}: timestamps not increasing")
        trans = np.array([ann_by_token[t]["translation"] for t in chain])
        if len(trans) > 1 and float(np.abs(trans - trans[0]).max()) > 1e-6:
            problems.append(
                f"track {inst['token']}: translation drift "
                f"{float(np.abs(trans - trans[0]).max()):.2e} (expected static)")

    return {
        "ok": not problems,
        "problems": problems,
        "n_instances": len(new_inst),
        "n_annotations": len(new_anns),
        "n_tracks": n_tracks,
        "n_total_annotations": len(nusc.sample_annotation),
        "sample_tokens_with_new": sorted({a["sample_token"] for a in new_anns}),
    }
