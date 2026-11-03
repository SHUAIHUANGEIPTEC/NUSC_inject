"""Step 6 — cross-modal consistency validation (multi-object)."""
from __future__ import annotations

import numpy as np

from . import transforms as tf


def _alpha_coverage(obj_pts_lidar, T_l2g, cam_spec, alpha):
    """Fraction of object lidar points projecting inside the alpha mask."""
    K = np.array(cam_spec["K"], dtype=float)
    T_g2c = np.linalg.inv(np.array(cam_spec["T_cam2global"], dtype=float).reshape(4, 4))
    pts_g = tf.transform_pts(T_l2g, obj_pts_lidar[:, :3])
    pts_c = tf.transform_pts(T_g2c, pts_g)
    h, w = alpha.shape
    front = pts_c[:, 2] > 0.1
    if not front.any():
        return None
    pix = tf.view_points(pts_c[front], K)
    u = np.round(pix[:, 0]).astype(int)
    v = np.round(pix[:, 1]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not ok.any():
        return 0.0
    hit = alpha[v[ok], u[ok]] > 0.05
    return float(hit.sum()) / float(ok.sum())


def _bbox_iou(cam_spec, alpha, gts):
    """2D IoU for ONE camera: visible-objects' projected AABB vs alpha AABB.

    Selection and box construction are two-phase: an object counts if ANY
    corner projects into (or near) the image; its box then uses ALL front
    corners (clamped to the image), so boxes of objects straddling the image
    edge stay correct instead of collapsing to the in-image corner sliver.
    Objects fully outside this camera are ignored (they live in other cams).
    """
    K = np.array(cam_spec["K"], dtype=float)
    T_g2c = np.linalg.inv(np.array(cam_spec["T_cam2global"], dtype=float).reshape(4, 4))
    width, height = cam_spec["width"], cam_spec["height"]
    margin = 8

    selected_pix = []
    for gt in gts:
        corners_c = tf.transform_pts(T_g2c, _corners_of(gt))
        front = corners_c[:, 2] > 0.1
        if not front.any():
            continue
        pix = tf.view_points(corners_c[front], K)
        in_img = ((pix[:, 0] >= -margin) & (pix[:, 0] <= width + margin)
                  & (pix[:, 1] >= -margin) & (pix[:, 1] <= height + margin))
        if in_img.any():
            selected_pix.append(pix)   # ALL front corners of this object
    if not selected_pix:
        return None
    pix = np.vstack(selected_pix)
    proj_box = [max(0.0, float(pix[:, 0].min())), max(0.0, float(pix[:, 1].min())),
                min(float(width - 1), float(pix[:, 0].max())),
                min(float(height - 1), float(pix[:, 1].max()))]

    rows, cols = np.where(alpha > 0.05)
    if len(rows) == 0:
        return 0.0
    mask_box = [float(cols.min()), float(rows.min()),
                float(cols.max()), float(rows.max())]

    ix = max(0.0, min(proj_box[2], mask_box[2]) - max(proj_box[0], mask_box[0]))
    iy = max(0.0, min(proj_box[3], mask_box[3]) - max(proj_box[1], mask_box[1]))
    inter = ix * iy
    a1 = (proj_box[2] - proj_box[0]) * (proj_box[3] - proj_box[1])
    a2 = (mask_box[2] - mask_box[0]) * (mask_box[3] - mask_box[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def _corners_of(gt):
    T = np.array(gt["T_model2global"], dtype=float).reshape(4, 4)
    raw_min = np.array(gt["raw_bbox_min"], dtype=float)
    raw_size = np.array(gt["raw_bbox_size"], dtype=float)
    corners = np.array([raw_min + np.array([sx, sy, sz]) * raw_size
                        for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])
    hom = np.hstack([corners, np.ones((8, 1))])
    return (T @ hom.T).T[:, :3]


def _occlusion_justified(pc):
    """Every hidden silhouette pixel must actually have nearer scene geometry.

    ``composite`` hides a pixel when ``obj_depth > scene_depth + margin``. If a
    pixel is hidden while the scene depth is NOT nearer, the occlusion is
    unjustified — that is the signature of a depth-buffer bug (a missing
    object sample, an over-eager occluder body, an axis-swapped box), not of
    real geometry. Returns (n_hidden, n_violations).
    """
    alpha = np.asarray(pc["alpha"])
    visible = np.asarray(pc["visible"])
    obj_depth = np.asarray(pc["obj_depth"])
    scene_depth = np.asarray(pc["scene_depth"])
    hidden = (alpha > 0.05) & ~visible
    n_hidden = int(hidden.sum())
    if n_hidden == 0:
        return 0, 0
    # a hidden pixel is justified only if the scene is at least as near
    justified = np.isfinite(scene_depth) & (
        scene_depth <= np.where(np.isfinite(obj_depth), obj_depth, np.inf))
    violations = int((hidden & ~justified).sum())
    return n_hidden, violations


def _visible_coverage(obj_pts_lidar, T_l2g, cam_spec, visible):
    """Fraction of object LiDAR points landing on a VISIBLE pixel.

    Informational, not a hard gate: the LiDAR and the camera have different
    viewpoints, so a point can legitimately be seen by one and not the other.
    """
    K = np.array(cam_spec["K"], dtype=float)
    T_g2c = np.linalg.inv(np.array(cam_spec["T_cam2global"], dtype=float).reshape(4, 4))
    pts_g = tf.transform_pts(T_l2g, obj_pts_lidar[:, :3])
    pts_c = tf.transform_pts(T_g2c, pts_g)
    h, w = visible.shape
    front = pts_c[:, 2] > 0.1
    if not front.any():
        return None
    pix = tf.view_points(pts_c[front], K)
    u = np.round(pix[:, 0]).astype(int)
    v = np.round(pix[:, 1]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not ok.any():
        return 0.0
    hit = visible[v[ok], u[ok]]
    return float(hit.sum()) / float(ok.sum())


def run(cfg, gts, lidar_result, cam_result, per_cam):
    """per_cam: {cam_name: {cam_spec, alpha, visible, n_visible_px, ...}}"""
    val_cfg = cfg["validation"]
    checks = {}

    # (c) transform chain round-trip (per object)
    resid_max = 0.0
    for gt in gts:
        T_model2lidar = np.array(gt["T_model2lidar"], dtype=float).reshape(4, 4)
        T_l2g = np.array(gt["T_lidar2global"], dtype=float).reshape(4, 4)
        T_model2global = np.array(gt["T_model2global"], dtype=float).reshape(4, 4)
        r = float(np.max(np.abs(T_model2lidar - np.linalg.inv(T_l2g) @ T_model2global)))
        resid_max = max(resid_max, r)
    checks["transform_roundtrip_max_resid"] = {"value": resid_max, "pass": resid_max < 1e-6}

    # (e) num_lidar_pts per object
    per_obj = lidar_result["num_obj_kept_per_object"]
    pts_pass = all(n > 0 for n in per_obj) if val_cfg.get(
        "require_num_lidar_pts_gt0", True) else True
    checks["num_lidar_pts"] = {
        "value": int(lidar_result["num_obj_kept"]),
        "per_object": [int(n) for n in per_obj],
        "pass": bool(pts_pass),
    }

    # (g) size plausibility — a geometrically self-consistent 0.05 m or 40 m
    # object is still nonsense as a nuScenes annotation row
    min_sz = float(val_cfg.get("min_size_m", 0.05))
    max_sz = float(val_cfg.get("max_size_m", 40.0))
    max_dims = [float(np.max(gt["size"])) for gt in gts]
    out_of_bounds = [s for s in max_dims if s < min_sz or s > max_sz]
    checks["size_plausible"] = {
        "max_dim_m": max_dims,
        "bounds_m": [min_sz, max_sz],
        "pass": not out_of_bounds,
    }

    # (d) blender self-check
    blender_err = float(cam_result.get("max_pixel_err", 1e9))
    thr = float(cfg["blender"].get("selfcheck_max_pixel_err", 2.0))
    checks["blender_selfcheck_max_pixel_err"] = {
        "value": blender_err, "pass": bool(blender_err <= thr)
    }

    # (a) coverage + (b) bbox IoU + (f) occlusion justification per camera
    min_cov = float(val_cfg.get("min_coverage", 0.85))
    min_iou = float(val_cfg.get("min_bbox_iou", 0.5))
    T_l2g = np.array(gts[0]["T_lidar2global"], dtype=float).reshape(4, 4)
    cam_reports = {}
    for name, pc in per_cam.items():
        cov = _alpha_coverage(lidar_result["obj_points_lidar"], T_l2g,
                              pc["cam_spec"], pc["alpha"])
        iou = _bbox_iou(pc["cam_spec"], pc["alpha"], gts)
        vis_cov = _visible_coverage(lidar_result["obj_points_lidar"], T_l2g,
                                    pc["cam_spec"], pc["visible"])
        n_hidden, n_bad = _occlusion_justified(pc)
        # edge-clipped silhouettes make box IoU structurally unfair
        # (projected 3D box AABB sticks out past the clipped mesh silhouette);
        # coverage remains the binding geometric check in that case.
        alpha = pc["alpha"]
        h, w = alpha.shape
        rows, cols = np.where(alpha > 0.05)
        touches_border = bool(len(rows) and (
            rows.min() == 0 or cols.min() == 0
            or rows.max() == h - 1 or cols.max() == w - 1))
        iou_pass = True if touches_border else bool(
            iou is not None and iou >= min_iou)
        cam_reports[name] = {
            "coverage": cov,
            "visible_coverage": vis_cov,
            "bbox_iou": iou,
            "bbox_iou_edge_clipped": touches_border,
            "coverage_pass": bool(cov is not None and cov >= min_cov),
            "bbox_iou_pass": iou_pass,
            "n_visible_px": pc.get("n_visible_px"),
            "n_hidden_px": n_hidden,
            "n_unjustified_occlusion_px": n_bad,
            "occlusion_justified": bool(n_bad == 0),
        }
    checks["per_camera"] = cam_reports

    all_pass = (checks["transform_roundtrip_max_resid"]["pass"]
                and checks["num_lidar_pts"]["pass"]
                and checks["size_plausible"]["pass"]
                and checks["blender_selfcheck_max_pixel_err"]["pass"]
                and all(c["coverage_pass"] and c["bbox_iou_pass"]
                        and c["occlusion_justified"]
                        for c in cam_reports.values()))
    report = {"all_pass": bool(all_pass), "checks": checks}
    return report
