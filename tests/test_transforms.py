"""Quaternion helpers and the nuScenes coordinate chain."""
from __future__ import annotations

import math

import numpy as np

from nusc_inject import transforms as tf


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0],
                     [math.sin(a), math.cos(a), 0.0],
                     [0.0, 0.0, 1.0]])


def test_quat_rot_roundtrip():
    for yaw in (0.0, 37.0, -90.0, 180.0, 271.0):
        R = _rz(yaw)
        q = tf.rot_to_quat(R)
        assert np.allclose(tf.quat_to_rot(q), R, atol=1e-12)

    # a rotation with no zero structure (all branches of Shepperd's method)
    R = tf.quat_to_rot([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(tf.quat_to_rot(tf.rot_to_quat(R)), R, atol=1e-12)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_quat_accepts_unnormalized_input():
    q = [2.0, 0.0, 0.0, 0.0]
    assert np.allclose(tf.quat_to_rot(q), np.eye(3), atol=1e-12)


def test_yaw_to_quat_matches_rz():
    for yaw_deg in (0.0, 30.0, -90.0, 150.0):
        assert np.allclose(
            tf.quat_to_rot(tf.yaw_to_quat(math.radians(yaw_deg))),
            _rz(yaw_deg), atol=1e-12)


def test_make_T_and_inverse():
    R = _rz(25.0)
    t = [3.0, -4.0, 1.5]
    T = tf.make_T(tf.rot_to_quat(R), t)
    assert np.allclose(T[:3, :3], R, atol=1e-12)
    assert np.allclose(T[:3, 3], t)
    assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0])

    pts = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    back = tf.transform_pts(np.linalg.inv(T), tf.transform_pts(T, pts))
    assert np.allclose(back, pts, atol=1e-12)


def test_T_to_quat_trans_roundtrip():
    T = tf.make_T(tf.yaw_to_quat(0.7), [1.0, 2.0, 3.0])
    q, t = tf.T_to_quat_trans(T)
    assert np.allclose(tf.make_T(q, t), T, atol=1e-12)


def test_transform_normals_rotates_but_does_not_translate():
    T = tf.make_T(tf.yaw_to_quat(math.pi / 2), [100.0, 100.0, 100.0])
    n = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(tf.transform_normals(T, n), [[0.0, 1.0, 0.0]], atol=1e-12)


def test_transform_normals_normalizes_output():
    T = np.eye(4)
    T[:3, :3] *= 3.0          # a scaled matrix must not leak into the normal
    out = tf.transform_normals(T, np.array([[2.0, 0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(out, axis=1), [1.0], atol=1e-12)


def test_view_points_uses_camera_depth():
    K = np.array([[100.0, 0.0, 320.0],
                  [0.0, 100.0, 240.0],
                  [0.0, 0.0, 1.0]])
    pts = np.array([[0.0, 0.0, 10.0], [1.0, -1.0, 5.0]])
    pix = tf.view_points(pts, K)
    assert np.allclose(pix[0], [320.0, 240.0], atol=1e-9)
    assert np.allclose(pix[1], [340.0, 220.0], atol=1e-9)


def test_lidar_to_cam_T_roundtrip(meta):
    lidar_sd, cam_sd = meta.sd_for_sample("t", "LIDAR_TOP"), {
        "calibrated_sensor_token": "cs_l", "ego_pose_token": "pose0"}
    T_l2g, T_g2c, K = tf.lidar_to_cam_T(lidar_sd, cam_sd, meta.calib, meta.poses)

    # T_l2g must be the ego pose composed with the sensor mount
    T_l2e = tf.make_T(meta.calib["cs_l"]["rotation"],
                      meta.calib["cs_l"]["translation"])
    T_e2g = tf.make_T(meta.poses["pose0"]["rotation"],
                      meta.poses["pose0"]["translation"])
    assert np.allclose(T_l2g, T_e2g @ T_l2e, atol=1e-12)

    # the lidar origin must land at the ego position expressed in the camera
    origin_cam = tf.transform_pts(T_g2c, tf.transform_pts(T_l2g, np.zeros((1, 3))))
    back = tf.transform_pts(np.linalg.inv(T_g2c @ T_l2g), origin_cam)
    assert np.allclose(back, np.zeros((1, 3)), atol=1e-9)
    assert K.shape == (3, 3)


def test_blender_cam_world_flips_opencv_axes():
    T = np.eye(4)
    assert np.allclose(tf.blender_cam_world(T), np.diag([1.0, -1.0, -1.0, 1.0]))


# ------------------------------------------------------------- euler angles


def test_euler_zero_is_identity():
    assert np.allclose(tf.euler_to_rot(0.0, 0.0, 0.0), np.eye(3))


def test_euler_yaw_only_matches_yaw_to_quat():
    for yaw in (0.0, 0.37, -1.2, math.pi):
        assert np.allclose(tf.euler_to_rot(yaw), tf.quat_to_rot(tf.yaw_to_quat(yaw)),
                           atol=1e-12)
        assert np.allclose(tf.euler_to_quat(yaw), tf.yaw_to_quat(yaw), atol=1e-12)


def test_euler_is_zyx_intrinsic():
    """Rz(yaw) @ Ry(pitch) @ Rx(roll) — verified by acting on basis vectors."""
    y, p, r = 0.4, -0.3, 0.9
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    assert np.allclose(tf.euler_to_rot(y, p, r), Rz @ Ry @ Rx, atol=1e-12)


def test_euler_rotation_is_orthonormal():
    for y, p, r in ((0.1, 0.2, 0.3), (-1.0, 1.4, 0.0), (0.0, 0.0, math.pi / 2)):
        R = tf.euler_to_rot(y, p, r)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_euler_pitch_tips_the_top_forward():
    """In a z-up frame with +x forward, +pitch about +Y tips +Z (up) toward
    +X (forward) and the nose (+x) toward -Z (down). Right-hand rule; see
    docs/model-input.md for the user-facing sign convention."""
    R = tf.euler_to_rot(0.0, math.radians(90.0), 0.0)
    assert np.allclose(R @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, -1.0], atol=1e-9)


def test_euler_roll_tilts_the_top_to_the_right():
    """+roll about +X tips +Z (up) toward -Y (right), again right-hand rule."""
    R = tf.euler_to_rot(0.0, 0.0, math.radians(90.0))
    assert np.allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, -1.0, 0.0], atol=1e-9)
