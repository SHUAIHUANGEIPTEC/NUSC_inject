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
from . import transforms as tf
from . import validate, visualize

log = logging.getLogger("pipeline")


def _sample_objects(objects, cfg, cache):
    """Per unique model: surface-sample once and cache (pts, normals, bbox, vsig, verts)."""
    for o in objects:
        path = o["model_cfg"]["path"]
        if path not in cache:
            pts, normals, bmin, bsize, vsig, verts = mesh_points.sample_mesh(
                path, int(cfg["mesh_points"]["n_dense"]))
            cache[path] = {"pts": pts, "normals": normals,
                           "bbox_min": bmin, "bbox_size": bsize, "vsig": vsig,
                           "verts": verts}
            log.info("sampled %s: raw bbox size=%s", Path(path).name,
                     np.round(bsize, 2))
    return cache


def _visible_px_per_object(lidar_result, gts, per_cam) -> list[int]:
    """Per object: max over cameras of its LiDAR points hitting a VISIBLE pixel.

    A target the depth test rejects in every camera scores 0 — the LiDAR
    still returns off it, yet the exported image shows nothing to detect, so
    it must not be drawn or annotated (the box would live only in metadata).
    LiDAR and camera viewpoints differ, so a point seen by one sensor and not
    the other is normal: this is a "hidden in ALL cameras" gate, not a
    per-camera visibility test.
    """
    pts = np.asarray(lidar_result["obj_points_lidar"])
    ids = np.asarray(lidar_result["obj_ids"])
    best = [0] * len(gts)
    if pts.size == 0 or not per_cam:
        return best
    T_l2g = np.asarray(gts[0]["T_lidar2global"], dtype=float).reshape(4, 4)
    pts_g = tf.transform_pts(T_l2g, pts[:, :3])
    for pc in per_cam.values():
        spec = pc["cam_spec"]
        K = np.asarray(spec["K"], dtype=float)
        T_g2c = np.linalg.inv(
            np.asarray(spec["T_cam2global"], dtype=float).reshape(4, 4))
        vis = np.asarray(pc["visible"])
        h, w = vis.shape
        pts_c = tf.transform_pts(T_g2c, pts_g)
        front = pts_c[:, 2] > 0.1
        if not front.any():
            continue
        pix = tf.view_points(pts_c[front], K)
        u = np.round(pix[:, 0]).astype(int)
        v = np.round(pix[:, 1]).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not ok.any():
            continue
        hit = vis[v[ok], u[ok]]
        idf = ids[front][ok]
        for i in range(len(gts)):
            m = idf == i
            if m.any():
                n = int(hit[m].sum())
                if n > best[i]:
                    best[i] = n
    return best


def _too_close_objects(gts, meta, sample, cfg) -> list[int]:
    """Indices of objects that nearly touch a camera able to frame them.

    A world-frozen track cannot dodge the ego: as the vehicle closes in, the
    target's bbox grows past the frame and its (often dark) back faces smear
    over the whole image — geometrically self-consistent, yet useless for
    detection. Pure geometry, so it runs BEFORE any render. Only cameras that
    actually frame the bbox count (frustum + margin), and ``min_obj_camera_
    dist_m = 0`` disables the gate entirely.
    """
    min_dist = float(cfg.get("validation", {}).get("min_obj_camera_dist_m", 0))
    if min_dist <= 0 or not gts:
        return []
    margin = int(cfg["camera_channel"].get("frustum_margin_px", 40))

    def _corners(gt):
        T = np.asarray(gt["T_model2global"], dtype=float).reshape(4, 4)
        rmin = np.asarray(gt["raw_bbox_min"], dtype=float)
        rsz = np.asarray(gt["raw_bbox_size"], dtype=float)
        c = np.array([rmin + np.array([sx, sy, sz]) * rsz
                      for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])
        hom = np.hstack([c, np.ones((8, 1))])
        return (T @ hom.T).T[:, :3]

    dead = []
    for i, gt in enumerate(gts):
        corners_g = _corners(gt)
        for channel in cfg["dataset"]["cam_channels"]:
            sd = meta.sd_for_sample(sample["token"], channel)
            T_g2c = camera_channel.global2cam(sd, meta)
            pts_c = tf.transform_pts(T_g2c, corners_g)
            front = pts_c[:, 2] > 0.1
            if not front.any():
                continue
            K = np.asarray(
                meta.calib[sd["calibrated_sensor_token"]]["camera_intrinsic"],
                dtype=float)
            pix = tf.view_points(pts_c[front], K)
            w, h = int(sd["width"]), int(sd["height"])
            inx = (pix[:, 0] >= -margin) & (pix[:, 0] <= w + margin)
            iny = (pix[:, 1] >= -margin) & (pix[:, 1] <= h + margin)
            if not (inx & iny).any():
                continue                      # this camera cannot see it
            if float(pts_c[front, 2].min()) < min_dist:
                dead.append(i)
                break
    return dead


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
                             raw_bbox=(m["bbox_min"], m["bbox_size"]),
                             model_verts=m["verts"])
        gts.append(gt)
    for i, gt in enumerate(gts):
        log.info("obj%d: pos=%s size=%s scale=%.5f ground_z=%.3f", i,
                 np.round(gt["translation"], 2), np.round(gt["size"], 2),
                 gt["scale"], gt["ground_z_lidar"])

    # ---- Step 1b: drop targets the ego has driven up to (render-free) ----
    # Later frames of a track can sit metres from a seeing camera: the bbox
    # smears across the image. Batch mode skips such frames (closing the
    # track) instead of burning a render on a frame no detector can use.
    too_close = _too_close_objects(gts, meta, sample, cfg)
    if too_close:
        if not drop_invisible:
            raise SystemExit(
                f"FATAL: object(s) {too_close} within min_obj_camera_dist_m "
                f"of a seeing camera (enable drop_invisible to skip frames)")
        log.warning("objects %s too close to a seeing camera (<%g m) — "
                    "skipped this frame, not drawn/annotated", too_close,
                    float(cfg["validation"].get("min_obj_camera_dist_m", 0)))
        dropped.extend(too_close)
        keep = [i for i in range(len(gts)) if i not in set(too_close)]
        if not keep:
            return {"status": "no_visible", "dropped": dropped,
                    "sample": sample, "out_dir": out_dir}
        objects = [objects[i] for i in keep]
        gts = [gts[i] for i in keep]
        track_keys = [track_keys[i] for i in keep]

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

    def _persist_lidar_gt():
        """Write lidar outputs + gt.json for the CURRENT lr/gts state.

        Runs once here and again after Step 4b drops a hidden object, so
        on-disk artefacts always match the annotated object set.
        """
        nusc_io.save_lidar_bin(out_dir / "lidar" / bin_name, lr["injected"])
        np.save(out_dir / "lidar" / "obj_mask.npy", lr["obj_mask"])
        np.save(out_dir / "lidar" / "obj_points_lidar.npy", lr["obj_points_lidar"])
        np.save(out_dir / "lidar" / "obj_ids.npy", lr["obj_ids"])
        log.info("LiDAR: scene %d + obj %d (per-object %s)",
                 lr["num_scene_kept"], lr["num_obj_kept"],
                 lr["num_obj_kept_per_object"])

        with open(out_dir / "gt.json", "w") as f:
            # root-relative model_path: run artefacts must not carry machine
            # paths either. gts themselves stay absolute (mesh_cache keys,
            # Blender job).
            root = cfg.get("_root")
            dump = [dict(g, model_path=annotation.rel_path(g["model_path"], root))
                    for g in gts]
            json.dump(dump if len(dump) > 1 else dump[0], f, indent=2)

    _persist_lidar_gt()

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

    # trimesh/Blender model-frame drift guard (per rendered object).
    # The bbox alone is not enough: an axis mix-up on a mesh with a symmetric
    # box leaves min/max unchanged. The surface signature (area + area-weighted
    # centroid) is tessellation-independent and the centroid moves under such a
    # permutation, so it catches what the box cannot.
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
                f"({ib['model_path']}): bbox drift {drift:.4f} m")
        sig_t = np.asarray(mesh_cache.get(gt["model_path"], {}).get("vsig", []),
                           dtype=float)
        sig_b = np.asarray(ib.get("vsig", []), dtype=float)
        if sig_t.size and sig_b.size:
            if not np.isclose(sig_t[0], sig_b[0], rtol=1e-4):
                raise SystemExit(
                    f"FATAL: trimesh/Blender surface area mismatch obj{i} "
                    f"({ib['model_path']}): {sig_t[0]:.4f} vs {sig_b[0]:.4f} m^2")
            centroid_drift = float(np.abs(sig_t[1:] - sig_b[1:]).max())
            if centroid_drift > 1e-3:
                raise SystemExit(
                    f"FATAL: trimesh/Blender model-frame mismatch obj{i} "
                    f"({ib['model_path']}): surface centroid drift "
                    f"{centroid_drift:.4f} m — likely an axis mix-up the bbox "
                    f"cannot see")

    # ---- Step 4: compositing ----
    comp_dir = out_dir / "composite"
    depth_dir = out_dir / "depth"
    comp_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    def _composite_pass() -> dict:
        """Composite the CURRENT payload into every usable camera.

        Re-entered after Step 4b drops a hidden object — a CPU-only redo;
        the Blender foreground of the survivors does not change.
        """
        cams = {}
        for c in cam_result["cams"]:
            name = c["name"]
            cam_spec = cam_result["cam_specs"][name]
            sd = meta.sd_for_sample(sample["token"], name)
            bg = cv2.cvtColor(cv2.imread(str(meta.sensor_path(sd))),
                              cv2.COLOR_BGR2RGB)
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
            # frustum margin may select cameras where the object barely misses
            # the frame (empty/tiny silhouette) — nothing to validate/replace
            min_alpha = int(cfg.get("validation", {}).get("min_alpha_px", 500))
            if info["n_alpha_px"] < min_alpha:
                log.info("  %s: alpha px=%d < %d, skip camera", name,
                         info["n_alpha_px"], min_alpha)
                continue
            cv2.imwrite(str(comp_dir / f"{name}.jpg"),
                        cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
            np.save(depth_dir / f"{name}_obj_depth.npy", obj_depth)
            np.save(depth_dir / f"{name}_scene_depth.npy", scene_depth)
            cams[name] = {"cam_spec": cam_spec, "alpha": alpha,
                          "visible": visible,
                          "n_visible_px": info["n_visible_px"],
                          "obj_depth": obj_depth, "scene_depth": scene_depth}
            log.info("  %s: visible=%d alpha=%d occluded=%d", name,
                     info["n_visible_px"], info["n_alpha_px"],
                     info["n_occluded_px"])
        return cams

    per_cam = _composite_pass()
    if not per_cam:
        # every camera fell under the min-alpha bar (a sliver of a thin pole
        # at range): in batch mode there is simply nothing to annotate this
        # frame — report it so the caller closes the tracks instead of
        # killing the whole run; single-sample mode keeps the hard failure.
        if drop_invisible:
            log.warning("no camera composited (all under min_alpha) -> "
                        "treat as no_camera")
            return {"status": "no_camera", "dropped": dropped,
                    "sample": sample, "out_dir": out_dir}
        raise SystemExit("FATAL: no camera composited")

    # ---- Step 4b: drop targets hidden in EVERY camera ----
    # The depth test can reject an object everywhere (it sits behind a lead
    # vehicle): LiDAR still hits it, but no pixel in any export image shows
    # it. Drawing/annotating it would ship a box only metadata knows, so under
    # drop_invisible such objects are dropped here and their track closes as
    # "invisible". Survivors get a CPU-only redo (lidar bin / gt.json /
    # composite) — Blender never re-renders, its foreground is unchanged.
    min_vis = int(cfg.get("validation", {}).get("min_visible_px", 0))
    if drop_invisible and min_vis > 0 and gts:
        hits = _visible_px_per_object(lr, gts, per_cam)
        dead = [i for i, n in enumerate(hits) if n < min_vis]
        if dead:
            log.warning("objects %s hidden in all cameras (visible-hit %s < %d)"
                        " — dropped, not drawn/annotated",
                        dead, [hits[i] for i in dead], min_vis)
            dropped.extend(dead)
            if len(dead) == len(gts):
                return {"status": "no_visible", "dropped": dropped,
                        "sample": sample, "out_dir": out_dir}
            keep = [i for i in range(len(gts)) if i not in set(dead)]
            objects = [objects[i] for i in keep]
            gts = [gts[i] for i in keep]
            track_keys = [track_keys[i] for i in keep]
            lr = lidar_channel.inject(scene_pts, _payload(), cfg,
                                      occluder_boxes=occ_boxes)
            _persist_lidar_gt()
            per_cam = _composite_pass()
            if not per_cam:
                # survivors all fall under the min-alpha bar in every camera:
                # nothing annotatable is left in this frame
                dropped.extend(keep)
                return {"status": "no_visible", "dropped": dropped,
                        "sample": sample, "out_dir": out_dir}

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
                                        track_keys=track_keys,
                                        root=cfg.get("_root"))
    with open(out_dir / "spec.json", "w") as f:
        json.dump(spec, f, indent=2)

    return {"status": "ok", "sample": sample, "gts": gts, "lidar_result": lr,
            "cam_result": cam_result, "per_cam": per_cam, "report": report,
            "spec": spec, "out_dir": out_dir, "dropped": dropped}
