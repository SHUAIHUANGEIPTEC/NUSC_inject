"""Spherical range-image binning (shared by the LiDAR channel and occluders)."""
from __future__ import annotations

import math

import numpy as np

from nusc_inject.range_image import dilate_min_map, spherical_bins

H, W = 64, 128
ELEV_MIN, ELEV_MAX = math.radians(-30.0), math.radians(10.0)


def _bins(pts):
    return spherical_bins(np.asarray(pts, dtype=float), H, W, ELEV_MIN, ELEV_MAX)


def test_center_of_fov_maps_to_middle_bin():
    row, col, rng, in_fov = _bins([[10.0, 0.0, 0.0]])
    assert int(row[0]) == 16          # elevation 0 sits a quarter way down
    assert int(col[0]) == 64          # azimuth 0 is the middle column
    assert rng[0] == 10.0 and bool(in_fov[0])


def test_azimuth_wraps_across_the_image():
    # +x -> az 0 -> col W/2; -x -> az pi -> col W (clipped to W-1)
    _, col, _, _ = _bins([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    assert [int(c) for c in col] == [64, W - 1, 96, 32]


def test_points_below_fov_are_out_of_fov_but_still_valid_bins():
    row, _, _, in_fov = _bins([[1.0, 0.0, -100.0]])
    assert not bool(in_fov[0])
    assert 0 <= int(row[0]) < H       # clipped, not negative or out of range


def test_points_above_fov_clip_to_first_row():
    row, _, _, in_fov = _bins([[1.0, 0.0, 100.0]])
    assert not bool(in_fov[0])
    assert int(row[0]) == 0


def test_range_is_euclidean_distance():
    _, _, rng, _ = _bins([[3.0, 4.0, 0.0]])
    assert rng[0] == 5.0


def test_bins_are_always_inside_the_image():
    rng_ = np.random.default_rng(0)
    pts = rng_.normal(size=(500, 3)) * 50.0
    row, col, _, _ = _bins(pts)
    assert row.min() >= 0 and row.max() < H
    assert col.min() >= 0 and col.max() < W
    assert np.issubdtype(row.dtype, np.integer)
    assert np.issubdtype(col.dtype, np.integer)


def test_higher_elevation_lands_in_a_higher_row():
    low = _bins([[10.0, 0.0, -1.0]])[0][0]
    high = _bins([[10.0, 0.0, 1.0]])[0][0]
    assert high < low                 # row 0 is top / highest elevation


# ---------------------------------------------------------------- dilate_min_map


def test_dilate_min_map_fills_the_neighbourhood_with_the_min():
    near = np.full((5, 5), np.inf)
    dilate_min_map(near, row=[2], col=[2], rng=[7.0], r_row=[1], r_col=[1])
    assert near[2, 2] == 7.0
    assert near[1, 1] == 7.0 and near[3, 3] == 7.0
    assert np.isinf(near[0, 0]) and np.isinf(near[4, 4])


def test_dilate_min_map_keeps_the_nearest_of_several_samples():
    near = np.full((5, 5), np.inf)
    dilate_min_map(near, row=[2, 2], col=[1, 3], rng=[9.0, 4.0],
                   r_row=[1, 1], r_col=[1, 1])
    assert near[2, 2] == 4.0          # nearer sample wins the overlap


def test_dilate_min_map_is_a_no_op_without_samples():
    near = np.full((3, 3), np.inf)
    out = dilate_min_map(near, row=[], col=[], rng=[], r_row=[], r_col=[])
    assert out is near and np.isinf(near).all()


def test_dilate_min_map_clips_at_the_image_border():
    near = np.full((4, 4), np.inf)
    dilate_min_map(near, row=[0], col=[0], rng=[3.0], r_row=[3], r_col=[3])
    assert near[0, 0] == 3.0 and near[3, 3] == 3.0     # window clipped, not raised

    near = np.full((4, 4), np.inf)
    dilate_min_map(near, row=[0], col=[0], rng=[3.0], r_row=[1], r_col=[1])
    assert near[1, 1] == 3.0
    assert np.isinf(near[2, 2])                         # radius 1 does not reach
