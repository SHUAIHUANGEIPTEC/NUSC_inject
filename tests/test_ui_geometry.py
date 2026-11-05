"""Back-projection geometry of the interactive placement UI.

``scripts/annotate_ui.py`` is a script, not a package module, so it is loaded
by path. Only the geometry is exercised — no HTTP, no Blender, no nuScenes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from nusc_inject import transforms as tf

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ui():
    spec = importlib.util.spec_from_file_location(
        "annotate_ui", ROOT / "scripts" / "annotate_ui.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ground(n=400, z=-1.7, extent=30.0):
    rng = np.random.default_rng(0)
    xy = rng.uniform(-extent, extent, size=(n, 2))
    return np.hstack([xy, np.full((n, 1), z)])


def _scene(K, T_cam2global, pts, T_l2e=None):
    """A minimal scene dict shaped like Workspace.scene()."""
    T_cam2global = np.asarray(T_cam2global, dtype=float)
    return {
        "sample": {"token": "s", "scene_token": "sc"},
        "pts": pts,
        "T_l2e": np.eye(4) if T_l2e is None else np.asarray(T_l2e, dtype=float),
        "T_e2g": np.eye(4),
        "T_l2g": np.eye(4),
        "cams": {"CAM_FRONT": {"K": np.asarray(K, dtype=float),
                               "T_cam2global": T_cam2global,
                               "width": 640, "height": 480, "path": None}},
        "ground_z": float(np.percentile(pts[:, 2], 15.0)),
    }


def _workspace(ui, scene):
    ws = object.__new__(ui.Workspace)          # skip __init__: no dataset load
    ws._scenes = {0: scene}
    ws._meshes = {}
    return ws


K = np.array([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]])

# A camera at the lidar origin looking along +X. Columns are the camera axes
# expressed in the lidar frame (x forward, y left, z up):
#   cam x(right) = -y, cam y(down) = -z, cam z(forward) = +x
R_CAM2LIDAR = np.array([[0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0]])
T_CAM2LIDAR = np.eye(4)
T_CAM2LIDAR[:3, :3] = R_CAM2LIDAR

# 2 m ahead, 0.5 m left, 1.7 m below -> camera frame x right, y down, z forward
P_LIDAR = np.array([2.0, 0.5, -1.7])
P_CAM = R_CAM2LIDAR.T @ P_LIDAR          # [-0.5, 1.7, 2.0]


def _pixel(p_cam):
    return (K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2],
            K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2])


def test_pick_recovers_a_known_ground_point(ui):
    """A pixel produced by projecting a ground point must pick that point back.

    This is what caught the original bug: the ray was rotated into global and
    straight back out again (R_g2l @ R_l2g), so the direction never changed.
    """
    ws = _workspace(ui, _scene(K, T_CAM2LIDAR, _ground()))
    u, v = _pixel(P_CAM)

    got = ws.pick(0, "CAM_FRONT", u, v)
    assert got["offset_ego"][0] == pytest.approx(P_LIDAR[0], abs=0.15)
    assert got["offset_ego"][1] == pytest.approx(P_LIDAR[1], abs=0.15)
    assert got["ground_z"] == pytest.approx(-1.7, abs=0.1)


def test_pick_scales_with_distance(ui):
    """Doubling the pixel offset from the principal point doubles the range."""
    ws = _workspace(ui, _scene(K, T_CAM2LIDAR, _ground()))
    near = ws.pick(0, "CAM_FRONT", 320.0, 320.0)["offset_ego"]
    far = ws.pick(0, "CAM_FRONT", 320.0, 265.0)["offset_ego"]
    # v further from the horizon -> closer; the ground point is straight ahead
    assert abs(near[0]) < abs(far[0])
    assert near[1] == pytest.approx(far[1], abs=1e-6)     # both centred


def test_pick_rejects_sky(ui):
    ws = _workspace(ui, _scene(K, T_CAM2LIDAR, _ground()))
    with pytest.raises(ValueError, match="天空"):
        ws.pick(0, "CAM_FRONT", 320.0, 0.0)      # above the horizon


def test_pick_respects_the_ego_frame(ui):
    """A yawed lidar mount must rotate the picked point into ego coordinates."""
    T_l2e = tf.make_T(tf.yaw_to_quat(np.radians(-90.0)), [0.0, 0.0, 1.8])
    ws = _workspace(ui, _scene(K, T_CAM2LIDAR, _ground(), T_l2e=T_l2e))
    u, v = _pixel(P_CAM)

    got = ws.pick(0, "CAM_FRONT", u, v)["offset_ego"]
    expected = tf.transform_pts(T_l2e, P_LIDAR[None, :])[0]
    assert got[0] == pytest.approx(expected[0], abs=0.15)
    assert got[1] == pytest.approx(expected[1], abs=0.15)
