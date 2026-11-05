"""Back-projection geometry of the interactive placement UI.

``scripts/annotate_ui.py`` is a script, not a package module, so it is loaded
by path. Only the geometry is exercised — no HTTP, no Blender, no nuScenes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conftest import StubMeta
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


# ------------------------------------------------- multi-object preview

_CFG = {
    "dataset": {"sample_index": 0, "lidar_channel": "LIDAR_TOP"},
    "model": {"path": "m.glb", "target_max_dim_m": 3.0, "albedo": 0.6,
              "intensity_gain_k": 200.0},
    "placement": {"strategy": "ego_offset", "offset_in_ego": [10.0, 0.0, 0.0],
                  "yaw_in_ego_deg": 0.0,
                  "ground": {"search_radius_m": 4.0, "z_percentile": 15.0}},
    "mesh_points": {"n_dense": 1000},
}


def _mesh_box(size=(1.0, 2.0, 3.0)):
    """A box-shaped mesh cache entry, so preview() never touches disk."""
    lo = -np.array(size, dtype=float) / 2.0
    corners = np.array([lo + np.array([sx, sy, sz]) * np.array(size)
                        for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])
    return {"pts": corners, "normals": np.zeros((8, 3)),
            "bbox_min": lo, "bbox_size": np.array(size, dtype=float),
            "verts": corners}


def _preview_ws(ui, scene, meta, meshes):
    ws = object.__new__(ui.Workspace)
    ws.cfg = _CFG
    ws.meta = meta
    ws._scenes = {0: scene}
    ws._meshes = dict(meshes)
    return ws


def _obj(id, offset_ego, yaw=0.0):
    return {"id": id, "model_path": "m.glb", "category": "vehicle.car",
            "target_max_dim_m": 3.0, "albedo": 0.6,
            "yaw": yaw, "pitch": 0.0, "roll": 0.0, "offset_ego": list(offset_ego)}


def test_preview_returns_one_wireframe_per_object(ui):
    ws = _preview_ws(ui, _scene(K, T_CAM2LIDAR, _ground()), StubMeta(),
                     {"m.glb": _mesh_box()})
    res = ws.preview({"sample_index": 0, "cam": "CAM_FRONT",
                      "objects": [_obj("a", [10.0, 0.0, 0.0]),
                                  _obj("b", [20.0, 2.0, 0.0], yaw=90.0)]})
    assert [o["id"] for o in res["objects"]] == ["a", "b"]
    for o in res["objects"]:
        assert len(o["corners_px"]) == 8
        assert len(o["in_front"]) == 8
        assert len(o["edges"]) == 12
        assert "size" in o["gt"]


def test_preview_objects_match_single_object_previews(ui):
    """Multi-object preview must not drift from previewing each object alone."""
    scene = _scene(K, T_CAM2LIDAR, _ground())
    meshes = {"m.glb": _mesh_box()}
    objs = [_obj("a", [10.0, 0.0, 0.0]), _obj("b", [20.0, 2.0, 0.0], yaw=90.0)]

    both = _preview_ws(ui, scene, StubMeta(), meshes)
    res = both.preview({"sample_index": 0, "cam": "CAM_FRONT", "objects": objs})

    for i, solo in enumerate(objs):
        one = _preview_ws(ui, scene, StubMeta(), meshes)
        got = one.preview({"sample_index": 0, "cam": "CAM_FRONT", "objects": [solo]})
        assert np.allclose(res["objects"][i]["corners_px"],
                           got["objects"][0]["corners_px"], atol=1e-6)
        assert res["objects"][i]["in_front"] == got["objects"][0]["in_front"]
        assert res["objects"][i]["gt"] == got["objects"][0]["gt"]


# --------------------------------------------------------- thumbnail job

def _thumb_case(size):
    bmin = np.array([1.0, -2.0, 0.5])
    bsize = np.array(size, dtype=float)
    corners = np.array([bmin + np.array([sx, sy, sz]) * bsize
                        for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])
    return bmin, bsize, corners


def test_thumb_job_centres_the_model_at_the_origin(ui):
    bmin, bsize, corners = _thumb_case((2.0, 4.0, 1.0))
    job = ui._thumb_job("m.glb", bmin, bsize, corners, "/tmp/t")
    T = np.array(job["objects"][0]["T_model2global"], dtype=float).reshape(4, 4)

    centre = bmin + bsize / 2.0
    assert np.allclose((T @ np.append(centre, 1.0))[:3], 0.0, atol=1e-9)
    # longest edge scaled to exactly 1 m
    assert np.allclose(T[:3, :3] @ bsize, bsize / bsize.max(), atol=1e-9)


def test_thumb_job_fills_the_frame(ui):
    """The camera distance must be solved per shape, not fixed at 2.2r.

    A fixed multiple of the bounding radius puts a long thin box at ~35% of the
    frame; the iterative solve lands near the 80% target for any shape.
    """
    for size in ((6.0, 2.4, 3.0), (0.45, 0.55, 1.75), (2.2, 0.4, 1.0)):
        bmin, bsize, corners = _thumb_case(size)
        job = ui._thumb_job("m.glb", bmin, bsize, corners, "/tmp/t")
        K = np.array(job["cams"][0]["K"], dtype=float)
        T_c2g = np.array(job["cams"][0]["T_cam2global"], dtype=float).reshape(4, 4)
        T = np.array(job["objects"][0]["T_model2global"], dtype=float).reshape(4, 4)

        uv = []
        for sx in (0, 1):
            for sy in (0, 1):
                for sz in (0, 1):
                    p = T @ np.append(bmin + np.array([sx, sy, sz]) * bsize, 1.0)
                    pc = np.linalg.inv(T_c2g) @ p
                    if pc[2] > 0:
                        q = K @ pc[:3]
                        uv.append(q[:2] / q[2])
        uv = np.array(uv)
        span = max(uv[:, 0].ptp(), uv[:, 1].ptp())
        assert 0.60 * 256 <= span <= 0.95 * 256, f"{size} span={span:.1f}"
