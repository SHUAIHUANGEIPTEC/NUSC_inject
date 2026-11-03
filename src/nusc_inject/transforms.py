"""Quaternion / rigid-transform utilities and the nuScenes coordinate chain.

Quaternion order follows nuScenes: (w, x, y, z).
"""
from __future__ import annotations

import math

import numpy as np


def quat_to_rot(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(R: np.ndarray):
    """Return (w, x, y, z) for a rotation matrix (Shepperd's method)."""
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def yaw_to_quat(yaw: float):
    """Rotation about +Z (nuScenes/Blender global up). Returns (w,x,y,z)."""
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def make_T(rotation, translation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(rotation)
    T[:3, 3] = [float(v) for v in translation]
    return T


def T_to_quat_trans(T: np.ndarray):
    return rot_to_quat(T[:3, :3]), T[:3, 3].copy()


def transform_pts(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts (N,3) -> (N,3) under 4x4 T."""
    return pts @ T[:3, :3].T + T[:3, 3]


def transform_normals(T: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Rotate (not translate) normals (N,3)."""
    n = normals @ T[:3, :3].T
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def view_points(pts_cam: np.ndarray, K: np.ndarray, normalize: bool = True) -> np.ndarray:
    """pts in CAMERA frame (N,3) -> (N,2) pixels via 3x3 K."""
    pts = np.asarray(pts_cam, dtype=float)
    proj = pts @ np.asarray(K, dtype=float).T
    if normalize:
        proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-8)
    return proj[:, :2]


def lidar_to_cam_T(lidar_sd, cam_sd, calib, poses):
    """Full nuScenes chain.

    Returns (T_l2g, T_g2c, K) such that
        p_cam = T_g2c @ T_l2g @ p_lidar
    Camera params: T_cam2global = inv(T_g2c), K = camera_intrinsic.
    Follows ood_to_camera.py:141-156.
    """
    T_l2e = make_T(calib[lidar_sd["calibrated_sensor_token"]]["rotation"],
                   calib[lidar_sd["calibrated_sensor_token"]]["translation"])
    pose_r = poses[lidar_sd["ego_pose_token"]]
    T_e2g = make_T(pose_r["rotation"], pose_r["translation"])
    T_l2g = T_e2g @ T_l2e

    cc = calib[cam_sd["calibrated_sensor_token"]]
    pose_c = poses[cam_sd["ego_pose_token"]]
    T_g2ce = np.linalg.inv(make_T(pose_c["rotation"], pose_c["translation"]))
    T_ce2c = np.linalg.inv(make_T(cc["rotation"], cc["translation"]))
    T_g2c = T_ce2c @ T_g2ce
    K = np.array(cc["camera_intrinsic"], dtype=float)
    return T_l2g, T_g2c, K


# OpenCV/nuScenes cam frame: x-right, y-down, z-forward.
# Blender cam frame:          x-right, y-up,   z-backward.
# p_opencv = diag(1,-1,-1) @ p_blender  ->  T_cam2global_opencv @ diag(1,-1,-1,1)
T_OPENCV2BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])


def blender_cam_world(T_cam2global_opencv: np.ndarray) -> np.ndarray:
    """matrix_world for a Blender camera matching the nuScenes/OpenCV pose."""
    return T_cam2global_opencv @ T_OPENCV2BLENDER
