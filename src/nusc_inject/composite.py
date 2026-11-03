"""Step 4 — depth occlusion + soft alpha compositing (multi-object).

obj_depth  : z-buffer over ALL objects' dense surface points projected into
             the camera (each object via its own T_model2global).
scene_depth: z-buffer of the original LiDAR points projected into the camera.
Rule       : visible = (alpha > thr) & (obj_depth <= scene_depth + margin)
"""
from __future__ import annotations

import cv2
import numpy as np

from . import occluders as occ_mod
from . import transforms as tf


def _zbuffer_depth(pts_cam, K, width, height):
    """Min-z per pixel. Returns (H,W) float32 with inf for empty pixels."""
    depth = np.full((height, width), np.inf, dtype=np.float32)
    z = pts_cam[:, 2]
    front = z > 0.1
    if not front.any():
        return depth
    pts = pts_cam[front]
    pix = tf.view_points(pts, K)
    u = np.round(pix[:, 0]).astype(np.int64)
    v = np.round(pix[:, 1]).astype(np.int64)
    ok = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[ok], v[ok], pts[ok, 2]
    np.minimum.at(depth, (v, u), z.astype(np.float32))
    return depth


def _fill_radius_px(K, fill_deg):
    """Pixel radius of an angular ``fill_deg`` footprint (pinhole, small angle)."""
    rad = np.radians(max(0.0, float(fill_deg)))
    if rad <= 0:
        return 0, 0
    K = np.asarray(K, dtype=float)
    return (max(0, int(np.ceil(rad * K[0, 0]))),
            max(0, int(np.ceil(rad * K[1, 1]))))


def _dilate_scene_depth(depth, radius_px):
    """Min-fill ``depth`` so each return covers a disc of ``radius_px`` pixels.

    Raw scene returns only land on the sensor's discrete beams, so a solid
    unannotated surface leaves most pixels empty and injected objects leak
    through it. Dilating the z-buffer by the beam footprint closes those gaps
    in the same way the range-image side does. Empty pixels stay ``inf``.
    """
    rx, ry = radius_px
    if rx <= 0 and ry <= 0:
        return depth
    rx, ry = max(1, rx), max(1, ry)
    big = 1e9
    work = np.where(np.isfinite(depth), depth, big).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))
    out = cv2.erode(work, k).astype(np.float64)
    out[out >= big * 0.5] = np.inf
    return out


def _fill_depth_holes(depth, mask, iters=4):
    """Fill un-sampled pixels inside ``mask`` from their nearest sampled neighbours.

    ``obj_depth`` is built from projected surface samples, so a pixel can have
    alpha > 0 (Blender saw geometry there) yet no sample landed on it. Left
    alone those pixels read as depth = +inf and get wrongly occluded; filling
    them keeps the depth test honest. Unfillable pixels stay ``inf``.
    """
    out = np.asarray(depth, dtype=np.float64).copy()
    k = np.ones((3, 3), np.float32)
    for _ in range(iters):
        holes = mask & ~np.isfinite(out)
        if not holes.any():
            break
        big = 1e9
        work = np.where(np.isfinite(out), out, big).astype(np.float32)
        neigh_min = cv2.erode(work, k)          # morphological min filter
        take = holes & (neigh_min < big * 0.5)
        if not take.any():
            break
        out[take] = neigh_min[take]
    return out


def composite_onto(name, fg_rgba, bg_rgb, objects, cam_spec, cfg, scene_pts_n5,
                   T_lidar2global=None, occluder_boxes=None):
    """Composite one camera over ALL objects.

    objects: list of {"pts_raw": (M,3), "gt": dict}
    scene_pts_n5: raw cloud in the LIDAR frame; T_lidar2global maps it to the
        global frame before the camera transform (lidar -> global -> camera).
    occluder_boxes: optional list of dense occluder bodies (from
        occluders.build_scene_occluder_boxes) — box faces rasterized per-pixel
        so near vehicles occlude injected objects even where LiDAR has no return.
    Returns (uint8 RGB, visible, alpha, obj_depth, scene_depth, info).
    """
    comp = cfg["composite"]
    alpha_thr = float(comp.get("alpha_threshold", 0.05))
    margin_base = float(comp.get("depth_margin_m", 0.2))
    margin_rel = float(comp.get("depth_margin_rel", 0.02))

    height, width = bg_rgb.shape[:2]
    K = np.array(cam_spec["K"], dtype=float)
    T_g2c = np.linalg.inv(np.array(cam_spec["T_cam2global"], dtype=float).reshape(4, 4))
    if T_lidar2global is None:
        T_l2g = np.eye(4)
    else:
        T_l2g = np.asarray(T_lidar2global, dtype=float).reshape(4, 4)
    T_l2c = T_g2c @ T_l2g  # lidar -> global -> camera

    obj_depth = np.full((height, width), np.inf, dtype=np.float32)
    for o in objects:
        T_model2global = np.array(o["gt"]["T_model2global"], dtype=float).reshape(4, 4)
        obj_cam = tf.transform_pts(T_g2c @ T_model2global, np.asarray(o["pts_raw"], dtype=float))
        obj_depth = np.minimum(obj_depth, _zbuffer_depth(obj_cam, K, width, height))

    scene_cam = tf.transform_pts(T_l2c, scene_pts_n5[:, :3])
    scene_depth = _zbuffer_depth(scene_cam, K, width, height)
    # beam-gap fill: raw returns only land on discrete scan lines, so a solid
    # unannotated surface would otherwise leak what it hides
    scene_depth = _dilate_scene_depth(
        scene_depth, _fill_radius_px(K, occ_mod.scene_gap_fill_deg(cfg)))
    if occluder_boxes:
        scene_depth = np.minimum(
            scene_depth, occ_mod.box_depth_raster(occluder_boxes, K, T_l2c, width, height))

    fg = fg_rgba.astype(np.float32) / 255.0
    if fg.shape[0] != height or fg.shape[1] != width:
        raise ValueError(f"foreground size {fg.shape[:2]} != background {(height, width)}")
    alpha = fg[:, :, 3] if fg.shape[2] == 4 else np.ones((height, width), np.float32)

    # Surface samples can miss a pixel that Blender still drew. Filling those
    # holes first keeps the depth test meaningful; whatever is still unknown
    # afterwards must not be silently erased — a missing sample is not
    # evidence that the scene is in front, so unknown object depth defers to
    # the Blender alpha (visible) rather than to the scene depth (occluded).
    obj_depth = _fill_depth_holes(obj_depth, alpha > alpha_thr)
    unknown_obj = ~np.isfinite(obj_depth)
    # The depth slack is not a constant: a fixed 0.5 m is 10% of the depth at
    # 5 m (a thin pedestrian leaks through) yet negligible at 60 m, while the
    # sparse-scene-depth error grows with range. `depth_margin_m` is the
    # constant term (calibration slack) and `depth_margin_rel` the range term.
    margin = margin_base + margin_rel * np.where(np.isfinite(obj_depth), obj_depth, 0.0)
    visible = (alpha > alpha_thr) & (
        unknown_obj | (obj_depth <= scene_depth + margin))

    bg = bg_rgb.astype(np.float32) / 255.0
    blended = fg[:, :, :3] * alpha[:, :, None] + bg * (1.0 - alpha[:, :, None])
    out = np.where(visible[:, :, None], blended, bg)

    info = {
        "name": name,
        "n_visible_px": int(visible.sum()),
        "n_alpha_px": int((alpha > alpha_thr).sum()),
        "n_occluded_px": int(((alpha > alpha_thr) & ~visible).sum()),
    }
    return (out * 255.0).astype(np.uint8), visible, alpha, obj_depth, scene_depth, info
