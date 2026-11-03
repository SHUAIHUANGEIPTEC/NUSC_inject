"""Anchor placement: ego-frame offsets, scaling, and the gt transform pair."""
from __future__ import annotations

import numpy as np
import pytest

from nusc_inject import anchor
from nusc_inject import transforms as tf

RAW_MIN = [0.0, 0.0, 0.0]
RAW_SIZE = [1.0, 2.0, 3.0]          # longest edge 3 m


def _cfg(scale=None, target=3.0, placement=None, sample_index=0):
    return {
        "dataset": {"sample_index": sample_index, "lidar_channel": "LIDAR_TOP"},
        "model": {"path": "m.glb", "scale": scale, "target_max_dim_m": target,
                  "albedo": 0.6, "intensity_gain_k": 200.0},
        "placement": placement or {
            "strategy": "ego_offset",
            "offset_in_ego": [12.0, 0.0, 0.0],
            "yaw_in_ego_deg": 0.0,
            "ground": {"search_radius_m": 4.0, "z_percentile": 15.0},
        },
    }


def test_resolve_scale_auto_fits_the_longest_edge():
    assert anchor.resolve_scale({"target_max_dim_m": 3.0}, RAW_SIZE) == 1.0
    assert anchor.resolve_scale({"target_max_dim_m": 6.0}, RAW_SIZE) == 2.0


def test_resolve_scale_explicit_overrides_target():
    assert anchor.resolve_scale({"scale": 0.25, "target_max_dim_m": 9.0},
                                RAW_SIZE) == 0.25


def test_normalize_T_moves_bbox_min_to_origin():
    T = anchor._normalize_T(np.array([1.0, -2.0, 0.5]))
    assert np.allclose(tf.transform_pts(T, np.array([[1.0, -2.0, 0.5]])), [[0, 0, 0]])


def test_build_gt_places_the_object_at_the_ego_offset(meta, ground_cloud):
    gt = anchor.build_gt(_cfg(), meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE))

    # LIDAR_TOP is mounted at Rz(-90), so ego +12 m forward is lidar +12 m left
    x, y, _ = gt["placement"]["offset_in_lidar_frame"]
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(12.0, abs=1e-9)
    assert gt["placement"]["offset_frame"] == "ego"

    # ground comes from the synthetic plane, not from the object
    assert gt["ground_z_lidar"] == pytest.approx(-1.7, abs=1e-9)

    # size scales uniformly, longest edge hits the target
    assert np.allclose(gt["size"], RAW_SIZE, atol=1e-9)
    assert gt["scale"] == pytest.approx(1.0)


def test_build_gt_keeps_the_two_transforms_consistent(meta, ground_cloud):
    gt = anchor.build_gt(_cfg(), meta, ground_cloud, raw_bbox=(RAW_MIN, RAW_SIZE))

    T_l2g = np.array(gt["T_lidar2global"]).reshape(4, 4)
    T_m2g = np.array(gt["T_model2global"]).reshape(4, 4)
    T_m2l = np.array(gt["T_model2lidar"]).reshape(4, 4)
    assert np.allclose(T_m2l, np.linalg.inv(T_l2g) @ T_m2g, atol=1e-9)

    # translation is the box centre, size is the scaled box extent
    center = np.array(RAW_MIN) + np.array(RAW_SIZE) / 2.0
    assert np.allclose(gt["translation"], tf.transform_pts(T_m2g, center[None, :])[0])


def test_build_gt_records_box_rotation_without_scale(meta, ground_cloud):
    gt = anchor.build_gt(_cfg(scale=2.0), meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE))
    R = tf.quat_to_rot(gt["rotation"])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.allclose(gt["size"], np.array(RAW_SIZE) * 2.0)


def test_yaw_in_ego_is_converted_to_the_lidar_mounting_frame(meta, ground_cloud):
    cfg = _cfg()
    cfg["placement"]["yaw_in_ego_deg"] = 30.0
    gt = anchor.build_gt(cfg, meta, ground_cloud, raw_bbox=(RAW_MIN, RAW_SIZE))
    # ego yaw 30 deg, mount -90 deg -> 120 deg in the lidar frame
    assert gt["placement"]["yaw_in_lidar_deg"] == pytest.approx(120.0, abs=1e-9)


def test_from_global_freezes_the_world_pose(meta, ground_cloud):
    base = anchor.build_gt(_cfg(), meta, ground_cloud, raw_bbox=(RAW_MIN, RAW_SIZE))
    frozen = np.array(base["T_model2global"], dtype=float).reshape(4, 4)

    cfg = _cfg(placement={"strategy": "frozen", "from_global": frozen.ravel().tolist()})
    gt = anchor.build_gt(cfg, meta, ground_cloud, raw_bbox=(RAW_MIN, RAW_SIZE))

    assert np.allclose(np.array(gt["T_model2global"]).reshape(4, 4), frozen, atol=1e-6)
    assert np.allclose(gt["translation"], base["translation"], atol=1e-6)
    assert np.allclose(gt["rotation"], base["rotation"], atol=1e-6)
    T_l2g = np.array(gt["T_lidar2global"]).reshape(4, 4)
    assert np.allclose(np.array(gt["T_model2lidar"]).reshape(4, 4),
                       np.linalg.inv(T_l2g) @ frozen, atol=1e-6)


def test_legacy_lidar_frame_offsets_still_work(meta, ground_cloud):
    cfg = _cfg(placement={"offset_in_lidar": [5.0, -3.0, 0.0],
                          "yaw_in_lidar_deg": 10.0})
    gt = anchor.build_gt(cfg, meta, ground_cloud, raw_bbox=(RAW_MIN, RAW_SIZE))
    assert gt["placement"]["offset_frame"] == "lidar"
    assert gt["placement"]["yaw_in_lidar_deg"] == pytest.approx(10.0, abs=1e-9)
    x, y, _ = gt["placement"]["offset_in_lidar_frame"]
    assert (x, y) == (pytest.approx(5.0), pytest.approx(-3.0))


def test_build_gt_requires_a_bbox(meta, ground_cloud):
    with pytest.raises(ValueError, match="raw_bbox"):
        anchor.build_gt(_cfg(), meta, ground_cloud)


def test_ground_falls_back_to_the_whole_cloud_when_locally_empty(meta, ground_cloud):
    """Placing far outside the cloud must not produce NaN/percentile of nothing."""
    cfg = _cfg()
    cfg["placement"]["offset_in_ego"] = [500.0, 500.0, 0.0]
    gt = anchor.build_gt(cfg, meta, ground_cloud, raw_bbox=(RAW_MIN, RAW_SIZE))
    assert gt["ground_z_lidar"] == pytest.approx(-1.7, abs=1e-9)
