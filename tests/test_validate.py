"""Step-6 validation: occlusion justification, size bounds, coverage metrics."""
from __future__ import annotations

import numpy as np

from nusc_inject import validate


def _cam_spec(w=4, h=4):
    K = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return {"name": "CAM_FRONT", "K": K,
            "T_cam2global": np.eye(4).ravel().tolist(),
            "width": w, "height": h}


def _gt(size=(1.0, 2.0, 3.0)):
    return {"size": list(size), "raw_bbox_min": [0.0, 0.0, 0.0],
            "raw_bbox_size": list(size),
            "T_model2global": np.eye(4).ravel().tolist(),
            "T_model2lidar": np.eye(4).ravel().tolist(),
            "T_lidar2global": np.eye(4).ravel().tolist()}


def _per_cam(hidden_justified=True, hidden_bad=False):
    alpha = np.zeros((4, 4), np.float32)
    visible = np.zeros((4, 4), bool)
    obj = np.full((4, 4), np.inf, np.float32)
    scene = np.full((4, 4), np.inf, np.float32)
    alpha[1:3, 1:3] = 1.0
    if hidden_justified:
        # hidden: scene is nearer than the object
        obj[1:3, 1:3] = 10.0
        scene[1:3, 1:3] = 5.0
    if hidden_bad:
        # hidden although the scene is NOT nearer -> unjustified
        obj[1:3, 1:3] = 5.0
        scene[1:3, 1:3] = 10.0
    if not hidden_justified and not hidden_bad:
        visible[1:3, 1:3] = True
        obj[1:3, 1:3] = 5.0
        scene[1:3, 1:3] = 10.0
    return {"cam_spec": _cam_spec(), "alpha": alpha, "visible": visible,
            "n_visible_px": int(visible.sum()),
            "obj_depth": obj, "scene_depth": scene}


# ------------------------------------------------------ occlusion justification


def test_occlusion_justified_passes_when_the_scene_is_nearer():
    n_hidden, n_bad = validate._occlusion_justified(_per_cam(hidden_justified=True))
    assert n_hidden == 4 and n_bad == 0


def test_occlusion_justified_flags_hiding_without_nearer_geometry():
    n_hidden, n_bad = validate._occlusion_justified(_per_cam(hidden_justified=False,
                                                             hidden_bad=True))
    assert n_hidden == 4 and n_bad == 4


def test_occlusion_justified_is_trivially_true_when_nothing_is_hidden():
    n_hidden, n_bad = validate._occlusion_justified(_per_cam(hidden_justified=False))
    assert n_hidden == 0 and n_bad == 0


# ------------------------------------------------------------------- size bounds


def _run(gts, val_cfg=None):
    cfg = {"validation": dict(val_cfg or {
        "min_coverage": 0.85, "min_bbox_iou": 0.5,
        "require_num_lidar_pts_gt0": False})}
    cfg["blender"] = {"selfcheck_max_pixel_err": 2.0}
    lidar = {"num_obj_kept": 1, "num_obj_kept_per_object": [1],
             "obj_points_lidar": np.zeros((1, 5))}
    cam = {"max_pixel_err": 0.01}
    per_cam = {"CAM_FRONT": _per_cam(hidden_justified=False)}
    return validate.run(cfg, gts, lidar, cam, per_cam)


def test_size_plausible_accepts_a_normal_object():
    rep = _run([_gt((1.0, 2.0, 3.0))])
    assert rep["checks"]["size_plausible"]["pass"]


def test_size_plausible_rejects_a_degenerate_object():
    rep = _run([_gt((0.001, 0.001, 0.001))])
    assert not rep["checks"]["size_plausible"]["pass"]
    assert not rep["all_pass"]


def test_size_plausible_rejects_an_absurd_object():
    rep = _run([_gt((500.0, 500.0, 500.0))])
    assert not rep["checks"]["size_plausible"]["pass"]


def test_unjustified_occlusion_fails_validation():
    cfg = {"validation": {"require_num_lidar_pts_gt0": False},
           "blender": {"selfcheck_max_pixel_err": 2.0}}
    lidar = {"num_obj_kept": 1, "num_obj_kept_per_object": [1],
             "obj_points_lidar": np.zeros((1, 5))}
    cam = {"max_pixel_err": 0.01}
    per_cam = {"CAM_FRONT": _per_cam(hidden_justified=False, hidden_bad=True)}
    rep = validate.run(cfg, [_gt()], lidar, cam, per_cam)
    assert rep["checks"]["per_camera"]["CAM_FRONT"]["n_unjustified_occlusion_px"] == 4
    assert not rep["all_pass"]


# ---------------------------------------------------------- visible coverage


def test_visible_coverage_counts_only_visible_pixels():
    visible = np.zeros((4, 4), bool)
    visible[0, 0] = True
    pts = np.array([[0.0, 0.0, 1.0]])          # projects to pixel (0,0) with K=I
    cov = validate._visible_coverage(pts, np.eye(4), _cam_spec(), visible)
    assert cov == 1.0

    pts_off = np.array([[2.0, 2.0, 1.0]])      # projects to (2,2), not visible
    cov = validate._visible_coverage(pts_off, np.eye(4), _cam_spec(), visible)
    assert cov == 0.0
