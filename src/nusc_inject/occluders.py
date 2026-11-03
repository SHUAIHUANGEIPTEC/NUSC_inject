"""Scene occluder bodies from nuScenes 3D annotations (sample_annotation).

The raw LiDAR cloud is sparse — glass / low-reflectance surfaces often have no
return — so a z-buffer built from scene points alone lets injected objects
"leak" through near vehicles in both channels. Each annotated 3D box is treated
as a coarse occluder body and rasterized into the occlusion z-buffer of the
LiDAR and camera channels (bodies are never written to the output cloud).

nuScenes ``sample_annotation`` boxes live in the GLOBAL frame; they are moved
into the LIDAR frame via ``inv(T_lidar2global)`` so they share the frame of the
raw point cloud.

Camera channel: each box's six faces are projected and filled per-pixel
(``box_depth_raster``), so the silhouette matches the box instead of a point
cloud. LiDAR channel: box surface points are binned on the spherical range
image and dilated by an adaptive radius (``occluder_scene_near``), filling the
gaps between sparse returns inside the body.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import transforms as tf
from .range_image import spherical_bins as _spherical_bins


def _face_grid(n_per_face):
    """Deterministic ~n_per_face samples in [0,1)^2 (grid, reproducible)."""
    k = max(2, int(round(np.sqrt(n_per_face))))
    a = (np.arange(k) + 0.5) / k
    u, v = np.meshgrid(a, a)
    return u.ravel(), v.ravel()


def box_surface_points(center, size, R, n_per_face, shrink=1.0):
    """Points on the six faces of one oriented box.

    center (3,) lidar frame, size (3,) nuScenes [w, l, h] (x, y, z),
    R (3,3) box->lidar rotation, shrink scales the box (<1 avoids over-occlusion).
    """
    s = np.asarray(size, dtype=float) * float(shrink)
    hx, hy, hz = s / 2.0
    u, v = _face_grid(n_per_face)
    faces = []
    for sign in (1.0, -1.0):
        # z faces (top / bottom): xy grid at z = +-hz
        faces.append(np.stack([(u * 2 - 1) * hx, (v * 2 - 1) * hy,
                               np.full_like(u, sign * hz)], axis=1))
        # y faces (front / back): xz grid at y = +-hy
        faces.append(np.stack([(u * 2 - 1) * hx, np.full_like(u, sign * hy),
                               (v * 2 - 1) * hz], axis=1))
        # x faces (left / right): yz grid at x = +-hx
        faces.append(np.stack([np.full_like(u, sign * hx), (u * 2 - 1) * hy,
                               (v * 2 - 1) * hz], axis=1))
    pts = np.vstack(faces)
    R = np.asarray(R, dtype=float)
    c = np.asarray(center, dtype=float)
    return pts @ R.T + c


def build_scene_occluder_boxes(meta, sample_token, cfg, T_lidar2global):
    """Occluder boxes (LIDAR frame) for one sample, from sample_annotation.

    Returns list of {"center": (3,), "size": (3,), "R": (3,3)} (box->lidar).
    cfg["scene_occluders"]: enabled (default true), scale (default 1.0),
    max_dist_m (default 60.0), min_visibility (default 1).
    """
    oc = cfg.get("scene_occluders") or {}
    if not oc.get("enabled", True):
        return []
    shrink = float(oc.get("scale", 1.0))
    dmax = float(oc.get("max_dist_m", 60.0))
    min_vis = int(oc.get("min_visibility", 1))

    T_g2l = np.linalg.inv(np.asarray(T_lidar2global, dtype=float).reshape(4, 4))
    R_g2l = T_g2l[:3, :3]
    t_g2l = T_g2l[:3, 3]

    boxes = []
    for a in meta.annotations_for_sample(sample_token):
        if int(a.get("visibility_token", "4")) < min_vis:
            continue
        c_l = R_g2l @ np.asarray(a["translation"], dtype=float) + t_g2l
        if float(np.hypot(c_l[0], c_l[1])) > dmax:
            continue
        boxes.append({
            "center": c_l,
            "size": np.asarray(a["size"], dtype=float) * shrink,
            "R": R_g2l @ tf.quat_to_rot(a["rotation"]),
        })
    return boxes


# camera-face corner quads (corner order: sx,sy,sz in {-1,+1}^3, x fastest)
_FACE_QUADS = (
    (0, 1, 3, 2),   # sz = -1 (bottom)
    (4, 5, 7, 6),   # sz = +1 (top)
    (0, 1, 5, 4),   # sy = -1 (front)
    (2, 3, 7, 6),   # sy = +1 (back)
    (0, 2, 6, 4),   # sx = -1 (left)
    (1, 3, 7, 5),   # sx = +1 (right)
)


def box_depth_raster(boxes, K, T_l2c, width, height):
    """Rasterize occluder boxes into a (H,W) depth buffer (inf = empty).

    Each of the six faces is projected and filled; a pixel gets the face's
    nearest corner depth (slightly conservative — bodies only ever occlude,
    never reveal). Boxes crossing the camera plane are skipped.
    """
    depth = np.full((height, width), np.inf, dtype=np.float32)
    K = np.asarray(K, dtype=float)
    T_l2c = np.asarray(T_l2c, dtype=float).reshape(4, 4)
    for b in boxes:
        s = np.asarray(b["size"], dtype=float) / 2.0
        corners = np.array([[sx * s[0], sy * s[1], sz * s[2]]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        cc = tf.transform_pts(T_l2c, corners @ np.asarray(b["R"]).T
                              + np.asarray(b["center"]))
        for quad in _FACE_QUADS:
            pc = cc[list(quad)]
            if (pc[:, 2] <= 0.1).any():
                continue  # face crosses/behind the camera plane
            pix = tf.view_points(pc, K)
            poly = np.round(pix).astype(np.int32)
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, poly, 1, lineType=cv2.LINE_AA)
            m = mask.astype(bool)
            if m.any():
                depth[m] = np.minimum(depth[m], float(pc[:, 2].min()))
    return depth


def occluder_scene_near(boxes, n_per_face, H, W, elev_min_deg, elev_max_deg,
                        range_max, spacing_m=0.15):
    """Min range per range-image bin from occluder bodies, with gap filling.

    Box surface points are sparse relative to the bin grid; each point dilates
    by an adaptive radius (``spacing_m`` of arc length at its own range) so the
    gaps between returns inside a body are filled. Returns (H,W) float64 with
    inf for bins without an occluder.
    """
    if not boxes:
        return np.full((H, W), np.inf, dtype=np.float64)
    elev_min = np.radians(elev_min_deg)
    elev_max = np.radians(elev_max_deg)
    near = np.full((H, W), np.inf, dtype=np.float64)

    d_az = 2.0 * np.pi / W              # azimuth rad per bin
    d_el = (elev_max - elev_min) / H    # elevation rad per bin
    # bucket points by range so one radius serves a whole bucket
    for b in boxes:
        pts = box_surface_points(b["center"], b["size"], b["R"], n_per_face)
        row, col, rng, in_fov = _spherical_bins(pts, H, W, elev_min, elev_max)
        ok = in_fov & (rng <= range_max) & (rng > 0.1)
        if not ok.any():
            continue
        row, col, rng = row[ok], col[ok], rng[ok]
        # adaptive bin radius from arc length at each point's range
        r_col = np.maximum(1, np.ceil(spacing_m / np.maximum(rng * d_az, 1e-6)).astype(np.int64))
        r_row = np.maximum(1, np.ceil(spacing_m / np.maximum(rng * d_el, 1e-6)).astype(np.int64))
        # bucket by radius so the window loop stays vectorized per bucket
        key = r_col * 10000 + r_row
        for k in np.unique(key):
            sel = key == k
            rr, cc, zz = row[sel], col[sel], rng[sel]
            rc, rrw = int(k // 10000), int(k % 10000)
            r0 = max(0, int(rr.min()) - rrw)
            r1 = min(H, int(rr.max()) + rrw + 1)
            c0 = max(0, int(cc.min()) - rc)
            c1 = min(W, int(cc.max()) + rc + 1)
            win = near[r0:r1, c0:c1]          # basic-slice VIEW (writes land in `near`)
            # per-point minimum over its own window
            rr0 = np.clip(rr - rrw - r0, 0, None)
            rr1 = np.clip(rr + rrw - r0 + 1, None, r1 - r0)
            cc0 = np.clip(cc - rc - c0, 0, None)
            cc1 = np.clip(cc + rc - c0 + 1, None, c1 - c0)
            for i in range(len(rr)):
                sub = win[rr0[i]:rr1[i], cc0[i]:cc1[i]]
                np.minimum(sub, zz[i], out=sub)
    return near
