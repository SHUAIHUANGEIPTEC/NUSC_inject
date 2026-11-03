"""LiDAR channel: two-sided range-image occlusion and Lambertian intensity."""
from __future__ import annotations

import math

import numpy as np

from nusc_inject import lidar_channel as lc

H, W = 64, 128
ELEV_MIN, ELEV_MAX = -30.0, 10.0


def _occlude(scene, obj, range_max=80.0, occluder_near=None):
    scene = np.asarray(scene, dtype=float).reshape(-1, 3)
    obj = np.asarray(obj, dtype=float).reshape(-1, 3)
    return lc.occlude(scene, obj, H, W, ELEV_MIN, ELEV_MAX, range_max,
                      occluder_near=occluder_near)


def _pt(az_deg, el_deg, r):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return [r * math.cos(el) * math.cos(az),
            r * math.cos(el) * math.sin(az),
            r * math.sin(el)]


def test_scene_points_are_never_deduplicated_against_each_other():
    """Two raw returns in the same bin are real data and must both survive."""
    scene = [_pt(0, 0, 10.0), _pt(0, 0, 12.0)]
    keep_scene, keep_obj = _occlude(scene, [])
    assert list(keep_scene) == [0, 1]
    assert list(keep_obj) == []


def test_nearer_object_drops_the_scene_returns_behind_it():
    scene = [_pt(0, 0, 10.0), _pt(0, 0, 12.0)]
    keep_scene, keep_obj = _occlude(scene, [_pt(0, 0, 8.0)])
    assert list(keep_scene) == []            # both share the object's bin
    assert list(keep_obj) == [0]


def test_farther_object_is_culled_but_scene_survives():
    scene = [_pt(0, 0, 10.0)]
    keep_scene, keep_obj = _occlude(scene, [_pt(0, 0, 15.0)])
    assert list(keep_scene) == [0]
    assert list(keep_obj) == []


def test_object_keeps_only_the_nearest_point_per_bin():
    obj = [_pt(0, 0, 9.0), _pt(0, 0, 8.0), _pt(0, 0, 8.5)]
    keep_scene, keep_obj = _occlude([], obj)
    assert list(keep_scene) == []
    assert list(keep_obj) == [1]             # only r=8.0


def test_object_points_outside_elevation_fov_are_dropped():
    keep_scene, keep_obj = _occlude([], [_pt(0, -60.0, 10.0)])
    assert list(keep_obj) == []


def test_object_points_beyond_range_max_are_dropped():
    _, keep_obj = _occlude([], [_pt(0, 0, 100.0)], range_max=80.0)
    assert list(keep_obj) == []


def test_scene_points_outside_fov_always_survive():
    # el=-60 is out of FOV; its row clips into the same bin as the object
    scene = [_pt(0, -60.0, 50.0)]
    keep_scene, _ = _occlude(scene, [_pt(0, 0, 5.0)])
    assert list(keep_scene) == [0]


def test_occluder_body_culls_object_it_hides_behind():
    """A dense body (no LiDAR return) must still occlude — the whole point of it."""
    near = np.full(H * W, np.inf)
    az = 0.0
    col = int((az + math.pi) / (2 * math.pi) * W)
    row = int((math.radians(ELEV_MAX) - 0.0)
              / (math.radians(ELEV_MAX) - math.radians(ELEV_MIN)) * H)
    near[row * W + col] = 5.0

    _, keep_obj = _occlude([], [_pt(0, 0, 10.0)], occluder_near=near)
    assert list(keep_obj) == []              # body at 5 m hides the 10 m object

    _, keep_obj = _occlude([], [_pt(0, 0, 3.0)], occluder_near=near)
    assert list(keep_obj) == [0]             # object in front of the body survives


def test_lambertian_intensity_faces_the_sensor():
    pts = np.array([[0.0, 0.0, -10.0]])      # directly below the sensor
    normals = np.array([[0.0, 0.0, 1.0]])    # faces up, towards origin -> lit
    I = lc.lambertian_intensity(pts, normals, albedo=0.6, gain_k=200.0)
    assert np.allclose(I, [120.0])


def test_lambertian_intensity_is_zero_for_back_facing_normals():
    pts = np.array([[0.0, 0.0, -10.0]])
    normals = np.array([[0.0, 0.0, -1.0]])   # faces away from the sensor
    assert lc.lambertian_intensity(pts, normals, 0.6, 200.0)[0] == 0.0


def test_lambertian_intensity_clips_to_255():
    pts = np.array([[0.0, 0.0, -10.0]])
    I = lc.lambertian_intensity(pts, np.array([[0.0, 0.0, 1.0]]),
                               albedo=1.0, gain_k=1000.0)
    assert I[0] == 255.0


def _cfg(n_lidar=0):
    return {
        "mesh_points": {"n_dense": 1000, "n_lidar": n_lidar},
        "lidar_channel": {
            "range_image": {"H": H, "W": W, "elev_min_deg": ELEV_MIN,
                            "elev_max_deg": ELEV_MAX},
            "range_max_m": 80.0,
            "out_format": "n5",
        },
    }


def _surface(r=10.0, n=64):
    """Blob of points around +x at range r, normals facing back at the sensor."""
    u = np.linspace(-0.2, 0.2, n)
    pts = np.array([[r, dy, dz] for dy in u for dz in u])
    normals = -pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts, normals


def test_inject_writes_object_points_with_intensity_and_ids():
    scene = np.hstack([np.array([_pt(0, 0, 40.0)] * 3), np.zeros((3, 2))])
    pts, normals = _surface()
    out = lc.inject(scene, [{"pts_raw": pts, "normals_raw": normals,
                             "gt": {"T_model2lidar": np.eye(4).ravel().tolist(),
                                    "albedo": 0.6, "intensity_gain_k": 200.0}}],
                    _cfg())

    assert out["injected"].shape[1] == 5
    assert out["obj_points_lidar"].shape[0] == out["num_obj_kept"] > 0
    assert out["num_obj_kept_per_object"] == [out["num_obj_kept"]]
    assert out["obj_mask"].sum() == out["num_obj_kept"]
    assert not out["obj_mask"][: out["num_scene_kept"]].any()
    assert np.all(out["obj_ids"] == 0)
    assert np.all(out["obj_points_lidar"][:, 3] >= 0.0)
    assert np.all(out["obj_points_lidar"][:, 3] <= 255.0)
    assert np.all(out["obj_points_lidar"][:, 4] == 0.0)      # ring column


def test_inject_respects_the_per_object_point_cap():
    scene = np.zeros((0, 5))
    pts, normals = _surface(n=40)
    out = lc.inject(scene, [{"pts_raw": pts, "normals_raw": normals,
                             "gt": {"T_model2lidar": np.eye(4).ravel().tolist()}}],
                    _cfg(n_lidar=5))
    assert out["num_obj_kept"] == 5
    assert out["obj_mask"].sum() == 5


def test_inject_is_deterministic_under_the_cap():
    scene = np.zeros((0, 5))
    pts, normals = _surface(n=40)
    obj = [{"pts_raw": pts, "normals_raw": normals,
            "gt": {"T_model2lidar": np.eye(4).ravel().tolist()}}]
    a = lc.inject(scene, obj, _cfg(n_lidar=7))["obj_points_lidar"]
    b = lc.inject(scene, obj, _cfg(n_lidar=7))["obj_points_lidar"]
    assert np.array_equal(a, b)


def test_inject_rejects_empty_object_list():
    import pytest
    with pytest.raises(ValueError, match="no objects to inject"):
        lc.inject(np.zeros((0, 5)), [], _cfg())


def test_inject_n4_output_keeps_only_four_columns():
    cfg = _cfg()
    cfg["lidar_channel"]["out_format"] = "n4"
    pts, normals = _surface()
    out = lc.inject(np.zeros((0, 5)),
                    [{"pts_raw": pts, "normals_raw": normals,
                      "gt": {"T_model2lidar": np.eye(4).ravel().tolist()}}], cfg)
    assert out["injected"].shape[1] == 4
