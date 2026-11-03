"""Compositing: per-pixel depth buffers and the occlusion rule."""
from __future__ import annotations

import numpy as np

from nusc_inject import composite

EYE = np.eye(3)
CFG = {"composite": {"alpha_threshold": 0.05, "depth_margin_m": 0.5}}


def _cam_spec(width=4, height=4):
    return {"name": "CAM_FRONT", "K": EYE.tolist(),
            "T_cam2global": np.eye(4).ravel().tolist(),
            "width": width, "height": height}


def _fg(width=4, height=4, patch=(slice(0, 2), slice(0, 2))):
    fg = np.zeros((height, width, 4), dtype=np.uint8)
    fg[patch[0], patch[1], :3] = 255
    fg[patch[0], patch[1], 3] = 255
    return fg


def _objects(z):
    """Four points covering the 2x2 foreground patch at depth ``z``."""
    pts = np.array([[0.0, 0.0, z], [z, 0.0, z], [0.0, z, z], [z, z, z]])
    return [{"pts_raw": pts,
             "gt": {"T_model2global": np.eye(4).ravel().tolist()}}]


def _scene(z):
    """LiDAR returns covering the same four pixels at depth ``z`` (N,5)."""
    pts = np.array([[0.0, 0.0, z], [z, 0.0, z], [0.0, z, z], [z, z, z]])
    return np.hstack([pts, np.zeros((4, 2))])


def test_zbuffer_keeps_the_nearest_point_per_pixel():
    depth = composite._zbuffer_depth(np.array([[0.0, 0.0, 5.0],
                                               [0.0, 0.0, 3.0]]), EYE, 4, 4)
    assert depth[0, 0] == 3.0
    assert np.isinf(depth[3, 3])


def test_zbuffer_ignores_points_behind_the_camera():
    depth = composite._zbuffer_depth(np.array([[0.0, 0.0, -5.0]]), EYE, 4, 4)
    assert np.isinf(depth).all()


def test_object_in_front_of_the_scene_is_visible():
    out, visible, alpha, obj_depth, scene_depth, info = composite.composite_onto(
        "CAM_FRONT", _fg(), np.zeros((4, 4, 3), np.uint8),
        _objects(5.0), _cam_spec(), CFG,
        scene_pts_n5=np.array([[0.0, 0.0, 10.0, 0.0, 0.0]]))

    assert visible[0, 0]
    assert info["n_visible_px"] == 4          # the 2x2 patch
    assert info["n_alpha_px"] == 4
    assert info["n_occluded_px"] == 0
    assert out[0, 0, 0] == 255                # foreground blended in


def test_object_behind_the_scene_is_masked_out():
    out, visible, _, _, scene_depth, info = composite.composite_onto(
        "CAM_FRONT", _fg(), np.zeros((4, 4, 3), np.uint8),
        _objects(20.0), _cam_spec(), CFG, scene_pts_n5=_scene(10.0))

    assert not visible[0, 0]
    assert info["n_visible_px"] == 0
    assert info["n_occluded_px"] == 4
    assert out[0, 0, 0] == 0                  # background kept


def test_depth_margin_tolerates_a_short_distance_error():
    """The margin exists so a coplanar LiDAR return does not erase the object."""
    cfg = {"composite": {"alpha_threshold": 0.05, "depth_margin_m": 1.0}}
    _, visible, _, _, _, _ = composite.composite_onto(
        "CAM_FRONT", _fg(), np.zeros((4, 4, 3), np.uint8),
        _objects(10.4), _cam_spec(), cfg,
        scene_pts_n5=np.array([[0.0, 0.0, 10.0, 0.0, 0.0]]))
    assert visible[0, 0]

    _, visible, _, _, _, _ = composite.composite_onto(
        "CAM_FRONT", _fg(), np.zeros((4, 4, 3), np.uint8),
        _objects(11.4), _cam_spec(), cfg,
        scene_pts_n5=np.array([[0.0, 0.0, 10.0, 0.0, 0.0]]))
    assert not visible[0, 0]


def test_transparent_foreground_never_occludes():
    _, visible, alpha, _, _, info = composite.composite_onto(
        "CAM_FRONT", _fg(patch=(slice(0, 1), slice(0, 1))),
        np.zeros((4, 4, 3), np.uint8),
        _objects(5.0), _cam_spec(), CFG,
        scene_pts_n5=np.zeros((0, 5)))
    assert info["n_alpha_px"] == 1
    assert alpha[1, 1] == 0.0
    assert not visible[1, 1]


def test_multiple_objects_share_one_depth_buffer():
    objects = _objects(20.0) + _objects(5.0)
    _, visible, _, obj_depth, _, _ = composite.composite_onto(
        "CAM_FRONT", _fg(), np.zeros((4, 4, 3), np.uint8),
        objects, _cam_spec(), CFG, scene_pts_n5=np.zeros((0, 5)))
    assert obj_depth[0, 0] == 5.0             # nearest object wins the buffer
    assert visible[0, 0]


def test_mismatched_foreground_size_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="foreground size"):
        composite.composite_onto(
            "CAM_FRONT", _fg(width=8, height=8), np.zeros((4, 4, 3), np.uint8),
            _objects(5.0), _cam_spec(), CFG, scene_pts_n5=np.zeros((0, 5)))
