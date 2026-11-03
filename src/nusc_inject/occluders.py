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
from .range_image import dilate_min_map, spherical_bins as _spherical_bins


def _face_grid(n_per_face):
    """Deterministic ~n_per_face samples in [0,1)^2 (grid, reproducible)."""
    k = max(2, int(round(np.sqrt(n_per_face))))
    a = (np.arange(k) + 0.5) / k
    u, v = np.meshgrid(a, a)
    return u.ravel(), v.ravel()


def _wlh_half_extents(size, scale=1.0):
    """Half-extents along the BOX axes for a nuScenes ``size = [w, l, h]``.

    nuScenes runs the box length along x and the width along y (devkit
    ``Box.corners``), so size[0]=w is the y extent and size[1]=l the x extent.
    Every consumer of an annotation box must go through here — reading
    size[0] as x silently rotates a car 90 deg in the ground plane.
    """
    s = np.asarray(size, dtype=float) * float(scale)
    return s[1] / 2.0, s[0] / 2.0, s[2] / 2.0


def box_surface_points(center, size, R, n_per_face, shrink=1.0):
    """Points on the six faces of one oriented box.

    center (3,) lidar frame, size (3,) nuScenes [w, l, h] (x extent = size[1]),
    R (3,3) box->lidar rotation, shrink scales the box (<1 avoids over-occlusion).
    """
    hx, hy, hz = _wlh_half_extents(size, shrink)
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


def _face_depth(pc, K, mask):
    """Per-pixel depth of a planar camera-frame quad.

    A box face is planar, so a ray ``z * K^{-1}[u, v, 1]`` meets it at
    ``z = d / (n · dir)``. Stamping the whole face with its NEAREST corner
    depth — the previous behaviour — over-occludes by up to the face's own
    depth extent on an oblique face (metres, not centimetres), which then
    forced ``depth_margin`` to be large enough to undo it.

    Returns ``(rows, cols, z)`` for the masked pixels, or ``None`` if the
    face is degenerate and the caller should fall back to the corner depth.
    """
    p0, p1, p2 = pc[0], pc[1], pc[2]
    n = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return None
    n = n / norm
    d = float(n @ p0)

    vs, us = np.where(mask)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    dirs = np.stack([(us.astype(np.float64) - cx) / fx,
                     (vs.astype(np.float64) - cy) / fy,
                     np.ones(len(us))], axis=1)
    denom = dirs @ n
    z = np.full(len(us), np.inf)
    ok = np.abs(denom) > 1e-9
    z[ok] = d / denom[ok]
    z[~(z > 0.1)] = np.inf               # behind the camera or non-finite
    return vs, us, z


def box_depth_raster(boxes, K, T_l2c, width, height):
    """Rasterize occluder boxes into a (H,W) depth buffer (inf = empty).

    Each of the six faces is projected and filled; a pixel gets that face's
    per-pixel planar depth (bodies only ever occlude, never reveal). Boxes
    crossing the camera plane are skipped.
    """
    depth = np.full((height, width), np.inf, dtype=np.float32)
    K = np.asarray(K, dtype=float)
    T_l2c = np.asarray(T_l2c, dtype=float).reshape(4, 4)
    for b in boxes:
        hx, hy, hz = _wlh_half_extents(b["size"])
        corners = np.array([[sx * hx, sy * hy, sz * hz]
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
            if not m.any():
                continue
            got = _face_depth(pc, K, m)
            if got is None:
                depth[m] = np.minimum(depth[m], float(pc[:, 2].min()))
            else:
                rows, cols, z = got
                np.minimum.at(depth, (rows, cols), z.astype(np.float32))
    return depth


def occluder_scene_near(boxes, n_per_face, H, W, elev_min_deg, elev_max_deg,
                        range_max, spacing_m=0.15):
    """Min range per range-image bin from occluder bodies, with gap filling.

    Box surface points are sparse relative to the bin grid; each point dilates
    by a constant **metric** footprint (``spacing_m`` of arc length at its own
    range) so the gaps between samples on a face are filled. Returns (H,W)
    float64 with inf for bins without an occluder.
    """
    if not boxes:
        return np.full((H, W), np.inf, dtype=np.float64)
    elev_min = np.radians(elev_min_deg)
    elev_max = np.radians(elev_max_deg)
    near = np.full((H, W), np.inf, dtype=np.float64)

    d_az = 2.0 * np.pi / W              # azimuth rad per bin
    d_el = (elev_max - elev_min) / H    # elevation rad per bin
    all_pts = np.vstack([box_surface_points(b["center"], b["size"], b["R"], n_per_face)
                         for b in boxes])
    row, col, rng, in_fov = _spherical_bins(all_pts, H, W, elev_min, elev_max)
    ok = in_fov & (rng <= range_max) & (rng > 0.1)
    row, col, rng = row[ok], col[ok], rng[ok]
    r_col = np.maximum(1, np.ceil(spacing_m / np.maximum(rng * d_az, 1e-6)).astype(np.int64))
    r_row = np.maximum(1, np.ceil(spacing_m / np.maximum(rng * d_el, 1e-6)).astype(np.int64))
    return dilate_min_map(near, row, col, rng, r_row, r_col)


def scene_gap_fill_deg(cfg) -> float:
    """Angular radius (degrees) used to dilate RAW scene returns.

    A solid unannotated surface (wall, building, canopy) only returns on the
    sensor's discrete beams — nuScenes LIDAR_TOP is a VLP-32, 32 beams spread
    over ~1.33 deg, while the range image has 512 rows at 0.081 deg. Dilating
    each return by a constant *metric* radius cannot bridge those gaps at
    every range (the gap grows with distance), so the fill radius is angular.

    The default 0.7 deg is about half the VLP-32 elevation beam spacing, which
    is the minimum that closes the gap; see ``tests/test_range_image.py``.
    """
    oc = cfg.get("scene_occluders") or {}
    return float(oc.get("scene_gap_fill_deg", 0.7))


def scene_near_from_returns(scene_pts, H, W, elev_min_deg, elev_max_deg,
                            range_max, fill_deg):
    """Min range per bin from raw scene returns, dilated by ``fill_deg``.

    Same output convention as ``occluder_scene_near``, so the two can be
    merged with ``np.minimum`` into one scene-side z-buffer.
    """
    elev_min = np.radians(elev_min_deg)
    elev_max = np.radians(elev_max_deg)
    near = np.full((H, W), np.inf, dtype=np.float64)
    if len(scene_pts) == 0:
        return near

    d_az = 2.0 * np.pi / W
    d_el = (elev_max - elev_min) / H
    row, col, rng, in_fov = _spherical_bins(np.asarray(scene_pts, dtype=float),
                                            H, W, elev_min, elev_max)
    ok = in_fov & (rng <= range_max) & (rng > 0.1)
    row, col, rng = row[ok], col[ok], rng[ok]
    # constant ANGULAR footprint -> constant radius in bins at every range
    rad = np.radians(float(fill_deg))
    r_col = np.maximum(1, np.ceil(rad / d_az).astype(np.int64))
    r_row = np.maximum(1, np.ceil(rad / d_el).astype(np.int64))
    return dilate_min_map(near, row, col, rng,
                          np.full(len(row), r_row),
                          np.full(len(row), r_col))
