"""Occluder bodies built from sample_annotation 3D boxes."""
from __future__ import annotations

import numpy as np
import pytest

from nusc_inject import occluders as occ

IDENTITY = np.eye(4)


def _ann(center, size=(2.0, 4.0, 1.5), visibility="4", rotation=None):
    return {
        "translation": list(center),
        "size": list(size),
        "rotation": rotation or [1.0, 0.0, 0.0, 0.0],
        "visibility_token": visibility,
    }


def test_face_grid_is_a_reproducible_square_grid():
    u, v = occ._face_grid(16)
    assert len(u) == len(v) == 16
    assert np.array_equal(u, occ._face_grid(16)[0])


def test_box_surface_points_follows_nuscenes_wlh_axis_order():
    """nuScenes size = [w, l, h]: the length runs along box-x, the width along box-y."""
    pts = occ.box_surface_points(np.zeros(3), [2.0, 4.6, 1.6], np.eye(3), n_per_face=4)
    assert np.abs(pts[:, 0]).max() == pytest.approx(2.3)   # x half-extent = l/2
    assert np.abs(pts[:, 1]).max() == pytest.approx(1.0)   # y half-extent = w/2
    assert np.abs(pts[:, 2]).max() == pytest.approx(0.8)   # z half-extent = h/2


def test_box_surface_points_land_on_the_box_surface():
    center = np.array([1.0, 2.0, 3.0])
    size = [2.0, 4.0, 6.0]              # nuScenes [w, l, h]
    pts = occ.box_surface_points(center, size, np.eye(3), n_per_face=4)

    assert len(pts) == 6 * 4
    local = pts - center
    half = np.array([2.0, 1.0, 3.0])    # x=l/2, y=w/2, z=h/2
    # every point sits on at least one face plane
    on_a_face = np.isclose(np.abs(local), half, atol=1e-9).any(axis=1)
    assert on_a_face.all()
    # and never outside the box
    assert (np.abs(local) <= half + 1e-9).all()
    assert np.allclose(pts.mean(axis=0), center, atol=1e-9)


def test_box_surface_points_honour_rotation_and_shrink():
    pts = occ.box_surface_points(np.zeros(3), [4.0, 4.0, 4.0],
                                 np.eye(3), n_per_face=4, shrink=0.5)
    assert np.abs(pts).max() == pytest.approx(1.0)

    # a 90 deg yaw about z maps box-x onto lidar-y
    yaw90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pts = occ.box_surface_points(np.zeros(3), [2.0, 4.0, 2.0], yaw90, n_per_face=4)
    # box half-extents (l/2, w/2, h/2) = (2, 1, 1); the yaw swaps x/y in lidar
    assert np.abs(pts[:, 0]).max() == pytest.approx(1.0)
    assert np.abs(pts[:, 1]).max() == pytest.approx(2.0)


def test_build_boxes_skips_far_and_low_visibility_annotations(meta):
    meta.annotations = [
        _ann([0.0, 0.0, 0.0]),               # kept
        _ann([200.0, 0.0, 0.0]),             # beyond max_dist_m
        _ann([1.0, 1.0, 0.0], visibility="1"),
    ]
    cfg = {"scene_occluders": {"enabled": True, "max_dist_m": 60.0,
                               "min_visibility": 2}}
    boxes = occ.build_scene_occluder_boxes(meta, "sample0", cfg, IDENTITY)
    assert len(boxes) == 1

    cfg["scene_occluders"]["min_visibility"] = 1
    assert len(occ.build_scene_occluder_boxes(meta, "sample0", cfg, IDENTITY)) == 2


def test_build_boxes_can_be_disabled(meta):
    meta.annotations = [_ann([0.0, 0.0, 0.0])]
    cfg = {"scene_occluders": {"enabled": False}}
    assert occ.build_scene_occluder_boxes(meta, "sample0", cfg, IDENTITY) == []


def test_build_boxes_moves_boxes_into_the_lidar_frame(meta):
    meta.annotations = [_ann([10.0, 0.0, 0.0])]
    # lidar sits at +5 m x with no rotation -> global 10 becomes lidar 5
    T_l2g = np.eye(4)
    T_l2g[0, 3] = 5.0
    cfg = {"scene_occluders": {"enabled": True}}
    boxes = occ.build_scene_occluder_boxes(meta, "sample0", cfg, T_l2g)
    assert boxes[0]["center"][0] == pytest.approx(5.0)


def test_box_depth_raster_fills_the_silhouette():
    box = {"center": np.array([0.0, 0.0, 5.0]),
           "size": np.array([2.0, 2.0, 2.0]),
           "R": np.eye(3)}
    K = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])
    depth = occ.box_depth_raster([box], K, np.eye(4), width=20, height=20)

    filled = np.isfinite(depth)
    assert filled.any()
    # nearest face is 1 m in front of the centre
    assert depth[filled].min() == pytest.approx(4.0)
    # the centre pixel is inside the box silhouette
    assert np.isfinite(depth[10, 10])


def test_box_depth_raster_uses_the_nuscenes_axis_order():
    """A car-sized box must cover the road footprint, not a 90 deg-rotated one."""
    box = {"center": np.array([0.0, 0.0, 5.0]),
           "size": np.array([2.0, 4.6, 1.6]),      # [w, l, h]
           "R": np.eye(3)}
    # fx = 10 at z = 5 -> 0.5 world units per pixel
    K = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])
    depth = occ.box_depth_raster([box], K, np.eye(4), width=20, height=20)
    rows, cols = np.where(np.isfinite(depth))
    assert len(rows) > 0
    # x extent = l = 4.6 spans more columns than the w = 2.0 spans rows
    assert (cols.max() - cols.min()) > (rows.max() - rows.min())


def test_box_depth_raster_skips_boxes_behind_the_camera():
    box = {"center": np.array([0.0, 0.0, -5.0]),
           "size": np.array([2.0, 2.0, 2.0]),
           "R": np.eye(3)}
    K = np.eye(3)
    depth = occ.box_depth_raster([box], K, np.eye(4), width=8, height=8)
    assert not np.isfinite(depth).any()


def test_occluder_scene_near_fills_bins_around_a_body():
    box = {"center": np.array([10.0, 0.0, 0.0]),
           "size": np.array([2.0, 2.0, 2.0]),
           "R": np.eye(3)}
    near = occ.occluder_scene_near([box], 16, H=64, W=128,
                                   elev_min_deg=-30.0, elev_max_deg=10.0,
                                   range_max=80.0, spacing_m=0.15)
    assert near.shape == (64, 128)
    assert np.isfinite(near).any()
    assert near[np.isfinite(near)].min() >= 9.0
    assert near[np.isfinite(near)].max() <= 11.0


def test_occluder_scene_near_is_empty_without_boxes():
    near = occ.occluder_scene_near([], 16, H=8, W=8, elev_min_deg=-30.0,
                                   elev_max_deg=10.0, range_max=80.0)
    assert np.isinf(near).all()
