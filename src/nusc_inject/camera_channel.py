"""Step 3 — camera channel driver (host side).

Builds job.json (multi-object), frustum-culls cameras against the union of
all object boxes, invokes Blender headless, reads result.json.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from . import transforms as tf


def _bbox_corners_global(gt: dict) -> np.ndarray:
    """8 box corners in the global frame (T acts on the RAW model frame)."""
    T = np.array(gt["T_model2global"], dtype=float).reshape(4, 4)
    raw_min = np.array(gt["raw_bbox_min"], dtype=float)
    raw_size = np.array(gt["raw_bbox_size"], dtype=float)
    corners = np.array([raw_min + np.array([sx, sy, sz]) * raw_size
                        for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])
    hom = np.hstack([corners, np.ones((8, 1))])
    return (T @ hom.T).T[:, :3]


def all_bbox_corners_global(gts) -> np.ndarray:
    return np.vstack([_bbox_corners_global(gt) for gt in gts])


def _cam_T(sd, meta):
    """T_global2camera for one camera sample_data record (ood_to_camera.py chain)."""
    cc = meta.calib[sd["calibrated_sensor_token"]]
    pose_c = meta.poses[sd["ego_pose_token"]]
    T_g2ce = np.linalg.inv(tf.make_T(pose_c["rotation"], pose_c["translation"]))
    T_ce2c = np.linalg.inv(tf.make_T(cc["rotation"], cc["translation"]))
    return T_ce2c @ T_g2ce


def _in_frustum(pts_cam, K, width, height, margin):
    if (pts_cam[:, 2] <= 0.1).all():
        return False
    pix = tf.view_points(pts_cam, K)
    inx = (pix[:, 0] >= -margin) & (pix[:, 0] <= width + margin)
    iny = (pix[:, 1] >= -margin) & (pix[:, 1] <= height + margin)
    front = pts_cam[:, 2] > 0.1
    return bool((inx & iny & front).any())


def build_job(cfg, meta, gts, out_dir: Path) -> dict:
    cams = []
    corners_g = all_bbox_corners_global(gts)
    margin = int(cfg["camera_channel"].get("frustum_margin_px", 40))

    for channel in cfg["dataset"]["cam_channels"]:
        sd = meta.sd_for_sample(gts[0]["sample_token"], channel)
        T_g2c = _cam_T(sd, meta)
        T_cam2global = np.linalg.inv(T_g2c)
        cc = meta.calib[sd["calibrated_sensor_token"]]
        K = np.array(cc["camera_intrinsic"], dtype=float)
        pts_cam = tf.transform_pts(T_g2c, corners_g)

        width, height = int(sd["width"]), int(sd["height"])
        if not _in_frustum(pts_cam, K, width, height, margin):
            continue
        cams.append({
            "name": channel,
            "K": K.tolist(),
            "T_cam2global": [float(v) for v in T_cam2global.reshape(-1)],
            "width": width,
            "height": height,
        })
    return cams


def run_blender(cfg, job: dict, out_dir: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    binary = cfg["blender"]["binary"]
    if not os.path.isabs(binary):
        binary = str(root / binary)
    if not os.path.exists(binary):
        raise FileNotFoundError(f"Blender binary not found: {binary} (run scripts/setup_blender.sh)")

    job_path = out_dir / "job.json"
    result_path = out_dir / "blender" / "result.json"
    os.makedirs(result_path.parent, exist_ok=True)
    job = dict(job)
    job["result_path"] = str(result_path)
    with open(job_path, "w") as f:
        json.dump(job, f, indent=2)

    script = str(root / "src" / "nusc_inject" / "blender_render.py")
    cmd = [binary, "--background", "--python", script, "--", "--job", str(job_path)]
    log_path = out_dir / "logs" / "blender_render.log"
    os.makedirs(log_path.parent, exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"Blender failed (rc={proc.returncode}), see {log_path}")

    with open(result_path) as f:
        return json.load(f)


def run(cfg, meta, gts, out_dir: Path) -> dict:
    """Frustum-cull cameras, render all objects in one Blender job.

    Returns the Blender result dict augmented with ``cam_specs`` (full K /
    T_cam2global entries, keyed by camera name).
    """
    cams = build_job(cfg, meta, gts, out_dir)
    if not cams:
        raise RuntimeError("objects not visible in any camera (frustum culling removed all)")
    job = {
        "mode": "render",
        # multi-object list; blender_render.py also accepts legacy single keys
        "objects": [{"model_path": gt["model_path"],
                     "T_model2global": gt["T_model2global"]} for gt in gts],
        "cams": cams,
        "output_dir": str(out_dir / "blender"),
        "engine": cfg["camera_channel"].get("engine", "CYCLES"),
        "cycles_device": cfg["camera_channel"].get("cycles_device", "CPU"),
        "cycles_samples": int(cfg["camera_channel"].get("cycles_samples", 64)),
        "lighting": cfg["camera_channel"].get("lighting", "fixed"),
        "selfcheck_max_pixel_err": float(cfg["blender"].get("selfcheck_max_pixel_err", 2.0)),
    }
    result = run_blender(cfg, job, out_dir)
    result["cam_specs"] = {c["name"]: c for c in cams}
    return result
