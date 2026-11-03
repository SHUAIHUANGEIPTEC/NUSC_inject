"""Step 4 — depth occlusion + soft alpha compositing (multi-object).

obj_depth  : z-buffer over ALL objects' dense surface points projected into
             the camera (each object via its own T_model2global).
scene_depth: z-buffer of the original LiDAR points projected into the camera.
Rule       : visible = (alpha > thr) & (obj_depth <= scene_depth + margin)
"""
from __future__ import annotations

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
    margin = float(comp.get("depth_margin_m", 0.5))

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
    if occluder_boxes:
        scene_depth = np.minimum(
            scene_depth, occ_mod.box_depth_raster(occluder_boxes, K, T_l2c, width, height))

    fg = fg_rgba.astype(np.float32) / 255.0
    if fg.shape[0] != height or fg.shape[1] != width:
        raise ValueError(f"foreground size {fg.shape[:2]} != background {(height, width)}")
    alpha = fg[:, :, 3] if fg.shape[2] == 4 else np.ones((height, width), np.float32)

    visible = (alpha > alpha_thr) & (obj_depth <= scene_depth + margin)

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
