"""Spherical range-image binning, shared by the LiDAR channel and occluders.

Both the LiDAR injection pass and the dense occluder bodies have to agree on
how a LIDAR-frame point maps to an (elevation row, azimuth column) cell — if
the two ever diverge, occlusion silently stops matching. One implementation,
one convention.
"""
from __future__ import annotations

import numpy as np


def spherical_bins(pts, H, W, elev_min, elev_max):
    """(row, col, range, in_fov) for LIDAR-frame points.

    ``elev_min`` / ``elev_max`` are radians. Rows and columns are clipped into
    the image; ``in_fov`` keeps the true elevation test so callers can decide
    to preserve out-of-FOV points instead of dropping them.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rng = np.linalg.norm(pts[:, :3], axis=1)
    az = np.arctan2(y, x)                     # [-pi, pi]
    el = np.arctan2(z, np.hypot(x, y))        # radians
    col = ((az + np.pi) / (2 * np.pi)) * W
    row = ((elev_max - el) / (elev_max - elev_min)) * H
    col = np.clip(col.astype(np.int64), 0, W - 1)
    row = np.clip(row.astype(np.int64), 0, H - 1)
    in_fov = (el >= elev_min) & (el <= elev_max)
    return row, col, rng, in_fov


def dilate_min_map(near, row, col, rng, r_row, r_col):
    """Min-fill ``near`` around each sample with its own (row, col) radius.

    ``near`` is a (H, W) float64 min-map (np.inf = empty) updated in place and
    also returned. Radii are in bins and are read per sample, so a caller can
    express either a constant metric footprint (occluder bodies) or a constant
    angular footprint (sensor beam gaps) by computing them differently.

    Buckets samples by their (r_col, r_row) pair so the inner loop stays a
    plain window write rather than a per-pixel scatter.
    """
    if len(row) == 0:
        return near
    H, W = near.shape
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    rng = np.asarray(rng, dtype=np.float64)
    r_col = np.maximum(1, np.asarray(r_col, dtype=np.int64))
    r_row = np.maximum(1, np.asarray(r_row, dtype=np.int64))
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
        rr0 = np.clip(rr - rrw - r0, 0, None)
        rr1 = np.clip(rr + rrw - r0 + 1, None, r1 - r0)
        cc0 = np.clip(cc - rc - c0, 0, None)
        cc1 = np.clip(cc + rc - c0 + 1, None, c1 - c0)
        for i in range(len(rr)):
            sub = win[rr0[i]:rr1[i], cc0[i]:cc1[i]]
            np.minimum(sub, zz[i], out=sub)
    return near
