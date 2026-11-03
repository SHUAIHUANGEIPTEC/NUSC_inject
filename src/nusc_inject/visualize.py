"""Debug visualization: point-on-image overlays and LiDAR BEV."""
from __future__ import annotations

import cv2
import numpy as np

from . import transforms as tf


def overlay_cam(cam_spec, bg_rgb, alpha, obj_pts_lidar, T_l2g, out_path):
    """Draw object lidar points on the image, colored by depth."""
    img = bg_rgb.copy()
    K = np.array(cam_spec["K"], dtype=float)
    T_g2c = np.linalg.inv(np.array(cam_spec["T_cam2global"], dtype=float).reshape(4, 4))
    pts_g = tf.transform_pts(T_l2g, obj_pts_lidar[:, :3])
    pts_c = tf.transform_pts(T_g2c, pts_g)
    h, w = img.shape[:2]
    front = pts_c[:, 2] > 0.1
    if front.any():
        pix = tf.view_points(pts_c[front], K)
        z = pts_c[front, 2]
        z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)
        for (u, v), t in zip(pix, z_norm):
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < w and 0 <= vi < h:
                color = (int(255 * (1 - t)), int(255 * t), 0)
                cv2.circle(img, (ui, vi), 2, color, -1)

    # alpha mask contour
    mask = (alpha > 0.05).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def lidar_bev(scene_pts, obj_pts, out_path, range_m=40.0, res=0.1):
    n = int(2 * range_m / res)
    img = np.full((n, n, 3), 30, dtype=np.uint8)

    def to_px(p):
        u = int((p[0] + range_m) / res)
        v = int((range_m - p[1]) / res)  # +y left -> up in BEV
        return u, v

    for p in scene_pts[:: max(1, len(scene_pts) // 20000)]:
        u, v = to_px(p)
        if 0 <= u < n and 0 <= v < n:
            img[v, u] = (120, 120, 120)
    for p in obj_pts:
        u, v = to_px(p)
        if 0 <= u < n and 0 <= v < n:
            img[v, u] = (0, 0, 255)

    cv2.imwrite(str(out_path), img)
