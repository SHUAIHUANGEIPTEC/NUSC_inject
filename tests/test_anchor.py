"""Anchor placement: ego-frame offsets, scaling, and the gt transform pair."""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import StubMeta
from nusc_inject import anchor
from nusc_inject import transforms as tf

RAW_MIN = [0.0, 0.0, 0.0]
RAW_SIZE = [1.0, 2.0, 3.0]          # longest edge 3 m
# the 8 bbox corners stand in for the mesh vertices of a box-shaped model
BOX_VERTS = np.array([np.array(RAW_MIN) + np.array([sx, sy, sz]) * np.array(RAW_SIZE)
                      for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])


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
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)

    # LIDAR_TOP is mounted at Rz(-90), so ego +12 m forward is lidar +12 m left
    x, y, _ = gt["placement"]["offset_in_lidar_frame"]
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(12.0, abs=1e-9)
    assert gt["placement"]["offset_frame"] == "ego"

    # ground comes from the synthetic plane, not from the object
    assert gt["ground_z_lidar"] == pytest.approx(-1.7, abs=1e-9)

    # size scales uniformly; gt['size'] is nuScenes [w, l, h] = [y, x, z] extents
    assert np.allclose(gt["size"], [RAW_SIZE[1], RAW_SIZE[0], RAW_SIZE[2]], atol=1e-9)
    assert gt["scale"] == pytest.approx(1.0)


def test_gt_size_is_in_nuscenes_wlh_order(meta, ground_cloud):
    """nuScenes sample_annotation.size = [w, l, h]: l along box-x, w along box-y."""
    raw_size = [1.0, 2.0, 3.0]          # model-frame x=1, y=2, z=3
    gt = anchor.build_gt(_cfg(scale=1.0), meta, ground_cloud,
                         raw_bbox=(RAW_MIN, raw_size), model_verts=BOX_VERTS)
    assert gt["size"] == [2.0, 1.0, 3.0]      # [w=y, l=x, h=z]

    # raw_bbox_size stays in the model frame — it feeds the corner maths, not
    # the nuScenes annotation row
    assert gt["raw_bbox_size"] == raw_size


def test_build_gt_keeps_the_two_transforms_consistent(meta, ground_cloud):
    gt = anchor.build_gt(_cfg(), meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)

    T_l2g = np.array(gt["T_lidar2global"]).reshape(4, 4)
    T_m2g = np.array(gt["T_model2global"]).reshape(4, 4)
    T_m2l = np.array(gt["T_model2lidar"]).reshape(4, 4)
    assert np.allclose(T_m2l, np.linalg.inv(T_l2g) @ T_m2g, atol=1e-9)

    # translation is the box centre, size is the scaled box extent
    center = np.array(RAW_MIN) + np.array(RAW_SIZE) / 2.0
    assert np.allclose(gt["translation"], tf.transform_pts(T_m2g, center[None, :])[0])


def test_build_gt_records_box_rotation_without_scale(meta, ground_cloud):
    gt = anchor.build_gt(_cfg(scale=2.0), meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)
    R = tf.quat_to_rot(gt["rotation"])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.allclose(gt["size"], [RAW_SIZE[1] * 2.0, RAW_SIZE[0] * 2.0,
                                    RAW_SIZE[2] * 2.0])


def test_yaw_in_ego_is_converted_to_the_lidar_mounting_frame(meta, ground_cloud):
    cfg = _cfg()
    cfg["placement"]["yaw_in_ego_deg"] = 30.0
    gt = anchor.build_gt(cfg, meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)
    # ego yaw 30 deg, mount -90 deg -> 120 deg in the lidar frame
    assert gt["placement"]["yaw_in_lidar_deg"] == pytest.approx(120.0, abs=1e-9)


def test_from_global_freezes_the_world_pose(meta, ground_cloud):
    base = anchor.build_gt(_cfg(), meta, ground_cloud,
                           raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)
    frozen = np.array(base["T_model2global"], dtype=float).reshape(4, 4)

    cfg = _cfg(placement={"strategy": "frozen", "from_global": frozen.ravel().tolist()})
    gt = anchor.build_gt(cfg, meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)

    assert np.allclose(np.array(gt["T_model2global"]).reshape(4, 4), frozen, atol=1e-6)
    assert np.allclose(gt["translation"], base["translation"], atol=1e-6)
    assert np.allclose(gt["rotation"], base["rotation"], atol=1e-6)
    T_l2g = np.array(gt["T_lidar2global"]).reshape(4, 4)
    assert np.allclose(np.array(gt["T_model2lidar"]).reshape(4, 4),
                       np.linalg.inv(T_l2g) @ frozen, atol=1e-6)


def test_legacy_lidar_frame_offsets_still_work(meta, ground_cloud):
    cfg = _cfg(placement={"offset_in_lidar": [5.0, -3.0, 0.0],
                          "yaw_in_lidar_deg": 10.0})
    gt = anchor.build_gt(cfg, meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)
    assert gt["placement"]["offset_frame"] == "lidar"
    assert gt["placement"]["yaw_in_lidar_deg"] == pytest.approx(10.0, abs=1e-9)
    x, y, _ = gt["placement"]["offset_in_lidar_frame"]
    assert (x, y) == (pytest.approx(5.0), pytest.approx(-3.0))


def test_build_gt_requires_a_bbox(meta, ground_cloud):
    with pytest.raises(ValueError, match="raw_bbox"):
        anchor.build_gt(_cfg(), meta, ground_cloud)


def test_build_gt_requires_model_verts(meta, ground_cloud):
    with pytest.raises(ValueError, match="model_verts"):
        anchor.build_gt(_cfg(), meta, ground_cloud, raw_bbox=(RAW_MIN, RAW_SIZE))


def test_ground_falls_back_to_the_whole_cloud_when_locally_empty(meta, ground_cloud):
    """Placing far outside the cloud must not produce NaN/percentile of nothing."""
    cfg = _cfg()
    cfg["placement"]["offset_in_ego"] = [500.0, 500.0, 0.0]
    gt = anchor.build_gt(cfg, meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)
    assert gt["ground_z_lidar"] == pytest.approx(-1.7, abs=1e-9)


# ------------------------------------------------ degenerate scale protection


def test_resolve_scale_rejects_a_zero_extent_model():
    with pytest.raises(ValueError, match="zero extent"):
        anchor.resolve_scale({"target_max_dim_m": 3.0}, [0.0, 0.0, 0.0])


def test_resolve_scale_rejects_a_non_positive_scale():
    with pytest.raises(ValueError, match="positive"):
        anchor.resolve_scale({"scale": 0.0}, RAW_SIZE)
    with pytest.raises(ValueError, match="positive"):
        anchor.resolve_scale({"scale": -1.0}, RAW_SIZE)


def test_resolve_scale_rejects_a_non_positive_target():
    with pytest.raises(ValueError, match="target_max_dim_m"):
        anchor.resolve_scale({"target_max_dim_m": 0.0}, RAW_SIZE)


def test_resolve_scale_rejects_a_range_valued_target_outside_batch():
    """run_batch.py samples the range first; anchor must not silently coerce."""
    with pytest.raises(ValueError, match="scalar"):
        anchor.resolve_scale({"target_max_dim_m": [2.5, 4.0]}, RAW_SIZE)


# -------------------------------------- from_global scale recovery (regression)


def test_scale_from_linear_recovers_a_uniform_scale():
    R = tf.quat_to_rot(tf.yaw_to_quat(0.7))
    assert anchor._scale_from_linear(R * 2.5) == pytest.approx(2.5)
    assert anchor._scale_from_linear(R * 0.001) == pytest.approx(0.001)


def test_scale_from_linear_rejects_a_non_uniform_linear_part():
    M = np.diag([2.0, 3.0, 2.0])
    with pytest.raises(ValueError, match="not a uniform scale"):
        anchor._scale_from_linear(M)


def test_scale_from_linear_rejects_a_singular_matrix():
    with pytest.raises(ValueError, match="non-positive"):
        anchor._scale_from_linear(np.zeros((3, 3)))


def test_from_global_recovers_scale_from_the_matrix_not_the_config(meta, ground_cloud):
    """A stale config scale must not silently contaminate rotation and size."""
    base = anchor.build_gt(_cfg(scale=2.0), meta, ground_cloud,
                           raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)
    frozen = np.array(base["T_model2global"], dtype=float).reshape(4, 4)

    # config now claims a DIFFERENT scale than the one baked into `frozen`
    cfg = _cfg(scale=9.9, placement={
        "strategy": "frozen", "from_global": frozen.ravel().tolist()})
    gt = anchor.build_gt(cfg, meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)

    assert gt["scale"] == pytest.approx(2.0)          # recovered, not 9.9
    assert np.allclose(gt["size"], [RAW_SIZE[1] * 2.0, RAW_SIZE[0] * 2.0,
                                    RAW_SIZE[2] * 2.0])
    assert np.allclose(gt["rotation"], base["rotation"], atol=1e-6)
    assert np.allclose(gt["translation"], base["translation"], atol=1e-6)


def test_from_global_rejects_a_matrix_that_is_not_scale_times_rotation(meta, ground_cloud):
    bad = np.eye(4)
    bad[0, 0], bad[1, 1] = 2.0, 3.0
    cfg = _cfg(placement={"from_global": bad.ravel().tolist()})
    with pytest.raises(ValueError, match="not a uniform scale"):
        anchor.build_gt(cfg, meta, ground_cloud,
                        raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)


# ------------------------------------------------- pitch / roll placement


def test_yaw_only_matches_the_scalar_formula_when_the_mount_is_pure_yaw(meta, ground_cloud):
    """Regression: with a pure-yaw sensor mount the matrix product must
    reproduce the old ``yaw_lidar = yaw_ego - mounting_yaw`` exactly."""
    cfg = _cfg()
    gt = anchor.build_gt(cfg, meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)
    T_m2l = np.array(gt["T_model2lidar"]).reshape(4, 4)

    lidar_sd = meta.sd_for_sample(gt["sample_token"], "LIDAR_TOP")
    cs = meta.calib[lidar_sd["calibrated_sensor_token"]]
    T_l2e = tf.make_T(cs["rotation"], cs["translation"])
    mounting_yaw = math.atan2(T_l2e[1, 0], T_l2e[0, 0])
    yaw_lidar_old = math.radians(cfg["placement"]["yaw_in_ego_deg"]) - mounting_yaw

    off = np.array([cfg["placement"]["offset_in_ego"]])
    p_l = tf.transform_pts(np.linalg.inv(T_l2e), off)[0]
    scale = gt["scale"]
    T_norm = np.eye(4)
    T_norm[:3, 3] = -np.array(RAW_MIN)
    T_pose_old = tf.make_T(tf.yaw_to_quat(yaw_lidar_old),
                           [float(p_l[0]), float(p_l[1]), gt["ground_z_lidar"]])
    T_m2l_old = T_pose_old @ np.diag([scale, scale, scale, 1.0]) @ T_norm

    assert np.allclose(T_m2l, T_m2l_old, atol=1e-12)
    assert gt["placement"]["yaw_in_lidar_deg"] == pytest.approx(math.degrees(yaw_lidar_old))


def test_yaw_only_stands_on_the_ego_vertical_with_a_tilted_mount(ground_cloud):
    """Real nuScenes calibration mounts LIDAR_TOP with ~1.4 deg of pitch.

    The old scalar formula left every injected object tilted by that much
    relative to the ground; composing the rotation as a matrix does not.
    """
    tilted = StubMeta(lidar_pitch_deg=1.4)
    cfg = _cfg()
    gt = anchor.build_gt(cfg, tilted, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)

    lidar_sd = tilted.sd_for_sample(gt["sample_token"], "LIDAR_TOP")
    cs = tilted.calib[lidar_sd["calibrated_sensor_token"]]
    T_l2e = tf.make_T(cs["rotation"], cs["translation"])

    R_m2l = np.array(gt["T_model2lidar"]).reshape(4, 4)[:3, :3] / gt["scale"]
    up_lidar = R_m2l @ np.array([0.0, 0.0, 1.0])
    ego_up_in_lidar = T_l2e[:3, :3].T @ np.array([0.0, 0.0, 1.0])

    assert np.allclose(up_lidar, ego_up_in_lidar, atol=1e-12)
    # the tilt is real, so this is NOT trivially the lidar z axis
    assert not np.allclose(ego_up_in_lidar, [0.0, 0.0, 1.0], atol=1e-3)


def test_pitch_is_composed_through_the_mounting_rotation(meta, ground_cloud):
    """+pitch must be rotated into the lidar frame, not simply applied there.

    LIDAR_TOP is mounted at ~-90 deg yaw vs ego, so a pitch about the ego Y
    axis does NOT stay a pitch about the lidar Y axis. Asserting the matrix
    product catches anyone who "simplifies" this back to a scalar addition.
    """
    pitch_ego = math.radians(30.0)
    cfg = _cfg(placement={"strategy": "ego_offset", "offset_in_ego": [12.0, 0.0, 0.0],
                          "yaw_in_ego_deg": 0.0, "pitch_in_ego_deg": 30.0,
                          "ground": {"search_radius_m": 4.0, "z_percentile": 15.0}})
    gt = anchor.build_gt(cfg, meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)

    lidar_sd = meta.sd_for_sample(gt["sample_token"], "LIDAR_TOP")
    cs = meta.calib[lidar_sd["calibrated_sensor_token"]]
    T_l2e = tf.make_T(cs["rotation"], cs["translation"])
    R_expected = T_l2e[:3, :3].T @ tf.euler_to_rot(0.0, pitch_ego, 0.0)

    R_actual = np.array(gt["T_model2lidar"]).reshape(4, 4)[:3, :3] / gt["scale"]
    assert np.allclose(R_actual, R_expected, atol=1e-12)

    # a naive "pitch straight in the lidar frame" must be wrong here
    R_naive = tf.euler_to_rot(0.0, pitch_ego, 0.0)
    assert not np.allclose(R_actual, R_naive, atol=1e-6), \
        "pitch was applied in the lidar frame instead of through T_l2e"


def test_roll_also_composes_through_the_mounting_rotation(meta, ground_cloud):
    cfg = _cfg(placement={"strategy": "ego_offset", "offset_in_ego": [12.0, 0.0, 0.0],
                          "yaw_in_ego_deg": 0.0, "roll_in_ego_deg": 25.0,
                          "ground": {"search_radius_m": 4.0, "z_percentile": 15.0}})
    gt = anchor.build_gt(cfg, meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)

    lidar_sd = meta.sd_for_sample(gt["sample_token"], "LIDAR_TOP")
    cs = meta.calib[lidar_sd["calibrated_sensor_token"]]
    T_l2e = tf.make_T(cs["rotation"], cs["translation"])
    R_expected = T_l2e[:3, :3].T @ tf.euler_to_rot(0.0, 0.0, math.radians(25.0))
    R_actual = np.array(gt["T_model2lidar"]).reshape(4, 4)[:3, :3] / gt["scale"]
    assert np.allclose(R_actual, R_expected, atol=1e-12)


def test_tilted_object_still_sits_on_the_ground(meta, ground_cloud):
    """Ground contact must be by the lowest support vertex, whatever the tilt."""
    cfg = _cfg(placement={"strategy": "ego_offset", "offset_in_ego": [12.0, 0.0, 0.0],
                          "yaw_in_ego_deg": 30.0, "pitch_in_ego_deg": 20.0,
                          "roll_in_ego_deg": -15.0,
                          "ground": {"search_radius_m": 4.0, "z_percentile": 15.0}})
    gt = anchor.build_gt(cfg, meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=BOX_VERTS)

    T_m2l = np.array(gt["T_model2lidar"]).reshape(4, 4)
    corners = np.array([np.array(RAW_MIN) + np.array([sx, sy, sz]) * np.array(RAW_SIZE)
                        for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])
    pts = (T_m2l[:3, :3] @ corners.T).T + T_m2l[:3, 3]
    # T_norm moved bbox-min to the origin, so the lowest vertex is exactly
    # where the object was dropped: ground_z.
    assert pts[:, 2].min() == pytest.approx(gt["ground_z_lidar"], abs=1e-9)


def test_ground_contact_is_by_mesh_vertices_not_bbox_corners(meta, ground_cloud):
    """A mesh whose bbox corners are empty must still touch the ground.

    Real models touch their bounding box at a few points only (a car at its
    tyres and bumpers), so the rotated bbox's lowest corner is usually empty
    space. Resting the *box* on the ground then leaves the *mesh* floating.
    Here the two feet are inset on the bottom face and the top reaches the
    corners, which keeps the bbox tight while leaving every bottom corner
    empty; a roll then brings an empty corner to the bottom.
    """
    support = np.array([
        [0.25, 0.5, 0.0], [0.75, 1.5, 0.0],     # feet: the only low geometry
        [0.0, 0.0, 3.0], [1.0, 0.0, 3.0],
        [0.0, 2.0, 3.0], [1.0, 2.0, 3.0],       # top corners -> tight bbox
    ])
    assert np.allclose(support.min(axis=0), RAW_MIN)
    assert np.allclose(support.max(axis=0), np.array(RAW_MIN) + np.array(RAW_SIZE))

    cfg = _cfg(placement={"strategy": "ego_offset", "offset_in_ego": [12.0, 0.0, 0.0],
                          "yaw_in_ego_deg": 0.0, "roll_in_ego_deg": 30.0,
                          "ground": {"search_radius_m": 4.0, "z_percentile": 15.0}})
    gt = anchor.build_gt(cfg, meta, ground_cloud,
                         raw_bbox=(RAW_MIN, RAW_SIZE), model_verts=support)

    T_m2l = np.array(gt["T_model2lidar"]).reshape(4, 4)
    ground = gt["ground_z_lidar"]
    feet = (T_m2l[:3, :3] @ support.T).T + T_m2l[:3, 3]
    corners = (T_m2l[:3, :3] @ BOX_VERTS.T).T + T_m2l[:3, 3]

    assert feet[:, 2].min() == pytest.approx(ground, abs=1e-9)
    # the box's lowest corner is empty space, so the box extends below the
    # mesh. That gap is exactly how far the mesh would FLOAT above the ground
    # if the box (rather than the mesh) were the thing rested on it — which is
    # what the old corner-based drop did.
    assert feet[:, 2].min() - corners[:, 2].min() > 0.1
