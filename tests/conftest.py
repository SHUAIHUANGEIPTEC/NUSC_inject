"""Shared fixtures: a minimal stand-in for NuScenesMini (no dataset needed)."""
from __future__ import annotations

import numpy as np
import pytest

from nusc_inject import transforms as tf


class StubMeta:
    """Implements only what anchor / occluders read from NuScenesMini."""

    def __init__(self, lidar_yaw_deg=-90.0, lidar_pitch_deg=0.0, ego_yaw_deg=15.0,
                 ego_translation=(100.0, 200.0, 0.0), annotations=None,
                 sample_token="sample0"):
        self.sample_rec = {"token": sample_token, "scene_token": "scene0"}
        self.calib = {
            "cs_l": {
                # real nuScenes calibration carries a small mounting pitch on
                # LIDAR_TOP; keep it configurable so tests can cover both
                "rotation": tf.euler_to_quat(np.radians(lidar_yaw_deg),
                                             np.radians(lidar_pitch_deg), 0.0),
                "translation": [0.0, 0.0, 1.8],
                "camera_intrinsic": [[100.0, 0.0, 320.0],
                                     [0.0, 100.0, 240.0],
                                     [0.0, 0.0, 1.0]],
            }
        }
        self.poses = {
            "pose0": {
                "rotation": tf.yaw_to_quat(np.radians(ego_yaw_deg)),
                "translation": list(ego_translation),
            }
        }
        self._lidar_sd = {
            "token": "sd_l",
            "sample_token": sample_token,
            "calibrated_sensor_token": "cs_l",
            "ego_pose_token": "pose0",
            "filename": "samples/LIDAR_TOP/x.pcd.bin",
            "is_key_frame": True,
            "width": 1600,
            "height": 900,
        }
        self.annotations = annotations or []

    def sample(self, index):
        return self.sample_rec

    def sd_for_sample(self, token, channel):
        return self._lidar_sd

    def annotations_for_sample(self, token):
        return self.annotations


@pytest.fixture
def meta():
    return StubMeta()


@pytest.fixture
def ground_cloud():
    """(N,3) LIDAR-frame cloud: flat ground plane at z = -1.7."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(-20.0, 20.0, size=(500, 2))
    return np.hstack([xy, np.full((500, 1), -1.7)])
