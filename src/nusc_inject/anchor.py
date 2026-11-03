"""Step 1 — anchor: choose object pose in the nuScenes global frame.

Single-source-of-truth output per object (gt.json entry):
  T_model2global (full 4x4 incl. normalization + uniform scale + pose)
  T_model2lidar  (same pose expressed in the LIDAR_TOP frame)
  translation/size/rotation (nuScenes-style box summary)

Model frame convention: Blender's imported frame (mesh_points normalizes
glb/gltf inputs), so both channels can apply T_* directly.
"""
from __future__ import annotations

import math

import numpy as np

from . import transforms as tf


def resolve_scale(model_cfg: dict, raw_size) -> float:
    """Uniform scale: explicit ``scale`` or auto from ``target_max_dim_m``.

    Degenerate inputs raise rather than producing inf/NaN transforms that
    would be written into ``gt.json`` and the annotation rows unnoticed.
    """
    s = model_cfg.get("scale")
    if s is None:
        target = model_cfg.get("target_max_dim_m", 3.0)
        if isinstance(target, (list, tuple)):
            raise ValueError(
                f"target_max_dim_m must be a scalar here, got {target!r} "
                f"(run_batch.py samples a range before calling)")
        target = float(target)
        if not (target > 0):
            raise ValueError(f"target_max_dim_m must be > 0, got {target}")
        longest = float(np.max(raw_size))
        if not (longest > 0):
            raise ValueError(f"model bbox has zero extent: {raw_size!r}")
        s = target / longest
    s = float(s)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"scale must be a finite positive number, got {s}")
    return s


def _scale_from_linear(M) -> float:
    """Uniform scale of ``M = R @ s`` (R orthonormal), with a consistency check.

    The 3x3 of a frozen ``T_model2global`` is exactly that product, so every
    column norm equals ``s``. Recovering ``s`` from the matrix — rather than
    re-deriving it from the config — means a frozen pose and its recorded
    ``gt['size']`` can never drift apart: if the config's scale no longer
    matches the one baked into ``from_global``, dividing by it would silently
    contaminate both the rotation and the size.
    """
    norms = np.linalg.norm(np.asarray(M, dtype=float), axis=0)
    s = float(norms.mean())
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"from_global 3x3 has non-positive scale: {norms}")
    if not np.allclose(norms, s, rtol=1e-5, atol=1e-9):
        raise ValueError(
            "from_global 3x3 is not a uniform scale times a rotation "
            f"(column norms {norms})")
    return s


def _normalize_T(raw_bbox_min: np.ndarray) -> np.ndarray:
    """Model frame: raw frame with bbox-min moved to origin."""
    T = np.eye(4)
    T[:3, 3] = -np.asarray(raw_bbox_min, dtype=float)
    return T


def build_gt(cfg, meta, lidar_pts, sample=None, model_cfg=None,
             placement_cfg=None, raw_bbox=None) -> dict:
    """Compose the placement transform and return the gt dict.

    sample: nuScenes sample dict (default: cfg dataset.sample_index).
    model_cfg / placement_cfg: per-object overrides (default: cfg["model"] / cfg["placement"]).
    raw_bbox: (bbox_min, bbox_size) in the Blender model frame (required).
    lidar_pts: (N,5) original scene cloud in the LIDAR_TOP frame (for ground z).
    """
    ds = cfg["dataset"]
    if sample is None:
        sample = meta.sample(int(ds["sample_index"]))
    model_cfg = model_cfg or cfg["model"]
    placement_cfg = placement_cfg or cfg["placement"]
    if raw_bbox is None:
        raise ValueError("raw_bbox=(bbox_min, bbox_size) is required")
    raw_min = np.asarray(raw_bbox[0], dtype=float)
    raw_size = np.asarray(raw_bbox[1], dtype=float)

    lidar_sd = meta.sd_for_sample(sample["token"], ds["lidar_channel"])
    cs = meta.calib[lidar_sd["calibrated_sensor_token"]]
    pose = meta.poses[lidar_sd["ego_pose_token"]]
    T_l2e = tf.make_T(cs["rotation"], cs["translation"])
    T_e2g = tf.make_T(pose["rotation"], pose["translation"])
    T_l2g = T_e2g @ T_l2e

    # --- placement (offsets in the EGO frame by default: x fwd, y left, z up;
    #     nuScenes LIDAR_TOP is mounted ~-90° yaw vs ego -> convert via T_l2e)
    pl = placement_cfg
    mounting_yaw = math.atan2(T_l2e[1, 0], T_l2e[0, 0])

    if "from_global" in pl:
        # frozen world pose (cross-frame tracking): same T_model2global every frame
        T_model2global = np.asarray(pl["from_global"], dtype=float).reshape(4, 4)
        T_model2lidar = np.linalg.inv(T_l2g) @ T_model2global
        # recover the scale from the frozen matrix, not from the config
        scale = _scale_from_linear(T_model2global[:3, :3])
        # 3x3 of T_model2global = R_l2g @ R_pose * scale (S/T_norm add no rotation)
        R_global_box = T_model2global[:3, :3] / scale
        R_pose = np.linalg.inv(T_l2g)[:3, :3] @ R_global_box
        p_anchor_l = tf.transform_pts(T_model2lidar, raw_min[None, :])[0]
        x, y, ground_z = (float(p_anchor_l[0]), float(p_anchor_l[1]),
                          float(p_anchor_l[2]))
        yaw_lidar = math.atan2(R_pose[1, 0], R_pose[0, 0])
    else:
        if "offset_in_ego" in pl:
            off_ego = np.array([float(v) for v in pl["offset_in_ego"]])
            p_anchor_l = tf.transform_pts(np.linalg.inv(T_l2e), off_ego[None, :])[0]
            yaw_lidar = math.radians(float(pl.get("yaw_in_ego_deg", 0.0))) - mounting_yaw
        else:  # legacy: offsets and yaw already in the lidar frame
            p_anchor_l = np.array([float(v) for v in pl["offset_in_lidar"]])
            yaw_lidar = math.radians(float(pl.get("yaw_in_lidar_deg", 0.0)))

        x, y = float(p_anchor_l[0]), float(p_anchor_l[1])

        g = pl.get("ground", {})
        radius = float(g.get("search_radius_m", 4.0))
        z_pct = float(g.get("z_percentile", 15.0))
        d2 = (lidar_pts[:, 0] - x) ** 2 + (lidar_pts[:, 1] - y) ** 2
        near = lidar_pts[d2 <= radius * radius]
        if len(near) < 10:
            near = lidar_pts
        ground_z = float(np.percentile(near[:, 2], z_pct))

        scale = resolve_scale(model_cfg, raw_size)
        S = np.eye(4)
        S[0, 0] = S[1, 1] = S[2, 2] = scale

        T_norm = _normalize_T(raw_min)  # bbox-min -> origin (object sits on z=0)
        T_pose = tf.make_T(tf.yaw_to_quat(yaw_lidar), [x, y, ground_z])

        T_model2lidar = T_pose @ S @ T_norm
        T_model2global = T_l2g @ T_model2lidar

    size = raw_size * scale
    center_raw = raw_min + raw_size / 2.0
    center_g = tf.transform_pts(T_model2global, center_raw[None, :])[0]

    # box orientation in the GLOBAL frame (nuScenes sample_annotation.rotation)
    R_global_box = T_model2global[:3, :3] / scale
    quat = tf.rot_to_quat(R_global_box)

    gt = {
        "category": model_cfg.get("category", "anomaly"),
        "albedo": float(model_cfg.get("albedo", 0.5)),
        "intensity_gain_k": float(model_cfg.get("intensity_gain_k", 200.0)),
        "translation": [float(v) for v in center_g],
        # nuScenes sample_annotation.size = [w, l, h]: l runs along the box x
        # axis, w along y, so the model-frame (x, y, z) extents map to [y, x, z].
        "size": [float(size[1]), float(size[0]), float(size[2])],
        "rotation": [float(v) for v in quat],
        "scale": scale,
        "model_path": model_cfg["path"],
        "raw_bbox_min": [float(v) for v in raw_min],
        "raw_bbox_size": [float(v) for v in raw_size],
        "T_model2global": [float(v) for v in T_model2global.reshape(-1)],
        "T_model2lidar": [float(v) for v in T_model2lidar.reshape(-1)],
        "T_lidar2global": [float(v) for v in T_l2g.reshape(-1)],
        "ground_z_lidar": ground_z,
        "placement": {
            "strategy": pl.get("strategy", "ego_offset"),
            "offset_in_lidar_frame": [x, y, ground_z],
            "offset_input": pl.get("offset_in_ego", pl.get("offset_in_lidar")),
            "offset_frame": "ego" if "offset_in_ego" in pl else "lidar",
            "yaw_in_lidar_deg": math.degrees(yaw_lidar),
        },
        "sample_token": sample["token"],
        "scene_token": sample["scene_token"],
        "sample_index": ds.get("sample_index"),
        "lidar_channel": ds["lidar_channel"],
    }
    return gt
