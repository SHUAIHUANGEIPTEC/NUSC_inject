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
