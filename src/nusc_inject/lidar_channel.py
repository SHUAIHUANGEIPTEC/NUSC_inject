"""Step 2 — LiDAR channel: surface points -> range-image occlusion -> intensity.

Multi-object capable: ALL objects are inserted into the SAME spherical z-buffer
as the scene, so objects can occlude each other / be occluded by the scene in
one pass. Same (azimuth, elevation) cell keeps only the nearest point.
"""
from __future__ import annotations

import numpy as np

from . import occluders as occ_mod
from . import transforms as tf
from .range_image import dilate_min_map
from .range_image import spherical_bins as _spherical_bins


def occlude(scene_pts, obj_pts, H, W, elev_min_deg, elev_max_deg, range_max,
            occluder_near=None, scene_fill_deg=0.7):
    """Two-sided occlusion in the spherical z-buffer.

    scene_pts (N,4+), obj_pts (M,4+) — both in the LIDAR frame.

    - scene points are dropped only when a STRICTLY NEARER object point
      occupies the same (row, col) bin (scene-vs-scene duplicates are real
      data and are never deduplicated);
    - object points keep only the nearest per bin, and only when strictly
      nearer than the nearest scene point in that bin;
    - the scene-side z-buffer dilates each raw return by ``scene_fill_deg``
      (an ANGULAR radius) so a solid unannotated surface occludes — raw
      returns only land on the sensor's discrete beams (VLP-32: 32 beams over
      512 range-image rows) and would otherwise leak ~94% of what they hide;
    - occluder_near (H*W, optional): dense body depth per bin (from
      occluders.occluder_scene_near) merged into the scene-side z-buffer so
      sparse/no-return surfaces still occlude injected points;
    - scene points outside the elevation FOV always survive.
    Returns (kept_scene_idx, kept_obj_idx).
    """
    elev_min = np.radians(elev_min_deg)
    elev_max = np.radians(elev_max_deg)
    n_scene = scene_pts.shape[0]
    n_bins = H * W

    all_pts = np.vstack([scene_pts[:, :3], obj_pts[:, :3]])
    row, col, rng, in_fov = _spherical_bins(all_pts, H, W, elev_min, elev_max)
    bin_id = row.astype(np.int64) * W + col.astype(np.int64)

    s_bins = bin_id[:n_scene]
    s_rng = rng[:n_scene]
    s_fov = in_fov[:n_scene]
    s_row, s_col = row[:n_scene], col[:n_scene]
    o_bins = bin_id[n_scene:]
    o_rng = rng[n_scene:]
    o_fov = in_fov[n_scene:] & (rng[n_scene:] <= range_max) & (rng[n_scene:] > 0.1)

    # scene-side z-buffer: raw returns, dilated so beam gaps do not leak
    scene_near2d = np.full((H, W), np.inf, dtype=np.float64)
    if s_fov.any():
        sr, sc, sz = s_row[s_fov], s_col[s_fov], s_rng[s_fov]
        rad = np.radians(max(0.0, float(scene_fill_deg)))
        if rad > 0:
            r_col = max(1, int(np.ceil(rad / (2.0 * np.pi / W))))
            r_row = max(1, int(np.ceil(rad / ((elev_max - elev_min) / H))))
            dilate_min_map(scene_near2d, sr, sc, sz,
                           np.full(len(sr), r_row), np.full(len(sr), r_col))
        else:
            np.minimum.at(scene_near2d, (sr, sc), sz)
    if occluder_near is not None:
        np.minimum(scene_near2d,
                   np.asarray(occluder_near, dtype=np.float64).reshape(H, W),
                   out=scene_near2d)
    scene_near = scene_near2d.reshape(-1)

    obj_near = np.full(n_bins, np.inf, dtype=np.float64)
    if o_fov.any():
        np.minimum.at(obj_near, o_bins[o_fov], o_rng[o_fov])

    # scene: keep unless a strictly nearer object shares the bin (out-of-FOV kept)
    drop_scene = s_fov & (obj_near[s_bins] < s_rng)
    keep_scene = np.where(~drop_scene)[0]

    # object: nearest per bin AND strictly nearer than nearest scene point.
    # The nearest test allows float ties — surface samples on a common shell
    # can land within 1 ULP of each other, and an exact `==` would drop them
    # nondeterministically.
    tol = 1e-9 * np.maximum(1.0, obj_near[o_bins])
    keep_obj_rel = np.where(
        o_fov & (o_rng <= obj_near[o_bins] + tol) & (o_rng < scene_near[o_bins])
    )[0]
    keep_obj = keep_obj_rel.astype(np.int64)

    return keep_scene, keep_obj


def lambertian_intensity(pts, normals, albedo, gain_k):
    """I = clip(k * rho * max(dot(-d, n), 0), 0, 255), d = p/||p||."""
    rng = np.linalg.norm(pts[:, :3], axis=1)
    d = pts[:, :3] / np.maximum(rng[:, None], 1e-9)
    cos_t = np.sum(-d * normals, axis=1)
    I = gain_k * albedo * np.maximum(cos_t, 0.0)
    return np.clip(I, 0.0, 255.0).astype(np.float32)


def inject(scene_pts_n5, objects, cfg, occluder_boxes=None):
    """Full Step-2 pipeline (multi-object).

    scene_pts_n5: (N,5) original cloud in LIDAR frame (x,y,z,intensity,ring)
    objects: list of {"pts_raw": (M,3), "normals_raw": (M,3), "gt": dict}
    occluder_boxes: optional list of dense occluder bodies (from
        occluders.build_scene_occluder_boxes) that take part in the range-image
        occlusion test only — they are never written to the output cloud.
    Returns dict with injected cloud pieces, per-object stats and obj_ids.
    """
    lc = cfg["lidar_channel"]
    ri = lc["range_image"]
    n_lidar_cap = int(cfg["mesh_points"].get("n_lidar", 0))

    all_pts, all_normals, all_ids = [], [], []
    for i, o in enumerate(objects):
        T = np.array(o["gt"]["T_model2lidar"], dtype=float).reshape(4, 4)
        pts = tf.transform_pts(T, np.asarray(o["pts_raw"], dtype=float))
        normals = tf.transform_normals(T, np.asarray(o["normals_raw"], dtype=float))
        # drop back-facing samples (no return on the far side)
        rng = np.linalg.norm(pts, axis=1)
        d = pts / np.maximum(rng[:, None], 1e-9)
        front = np.sum(-d * normals, axis=1) > 0
        all_pts.append(pts[front])
        all_normals.append(normals[front])
        all_ids.append(np.full(int(front.sum()), i, dtype=np.int32))
    if not all_pts:
        raise ValueError("no objects to inject")

    obj_pts = np.vstack(all_pts)
    obj_normals = np.vstack(all_normals)
    obj_ids = np.concatenate(all_ids)

    obj_col = np.zeros((obj_pts.shape[0], 5), dtype=np.float32)
    obj_col[:, :3] = obj_pts

    occl_near = None
    if occluder_boxes:
        oc = cfg.get("scene_occluders") or {}
        occl_near = occ_mod.occluder_scene_near(
            occluder_boxes, int(oc.get("n_per_face", 64)),
            int(ri["H"]), int(ri["W"]),
            float(ri["elev_min_deg"]), float(ri["elev_max_deg"]),
            float(lc.get("range_max_m", 80.0)),
            spacing_m=float(oc.get("spacing_m", 0.15)),
        )
    k_scene, k_obj = occlude(
        scene_pts_n5, obj_col,
        int(ri["H"]), int(ri["W"]),
        float(ri["elev_min_deg"]), float(ri["elev_max_deg"]),
        float(lc.get("range_max_m", 80.0)),
        occluder_near=occl_near,
        scene_fill_deg=occ_mod.scene_gap_fill_deg(cfg),
    )

    kept_ids = obj_ids[k_obj]
    kept_obj = obj_col[k_obj].copy()
    kept_normals = obj_normals[k_obj]
    # per-object Lambertian intensity (albedo/gain come from each object's gt)
    kept_I = np.zeros(kept_obj.shape[0], dtype=np.float32)
    for i, o in enumerate(objects):
        m = kept_ids == i
        if not m.any():
            continue
        g = o["gt"]
        kept_I[m] = lambertian_intensity(
            kept_obj[m], kept_normals[m],
            float(g.get("albedo", 0.5)), float(g.get("intensity_gain_k", 200.0)),
        )
    kept_obj[:, 3] = kept_I
    kept_obj[:, 4] = 0.0  # ring

    # per-object subsample cap (deterministic)
    if n_lidar_cap > 0:
        sel = []
        for i in range(len(objects)):
            idx = np.where(kept_ids == i)[0]
            if idx.size > n_lidar_cap:
                rngp = np.random.default_rng(i)
                sel.append(rngp.choice(idx, n_lidar_cap, replace=False))
        if sel:
            sel = np.sort(np.concatenate(sel))
            kept_obj = kept_obj[sel]
            kept_normals = kept_normals[sel]
            kept_ids = kept_ids[sel]

    kept_scene = scene_pts_n5[k_scene].copy()

    out_fmt = lc.get("out_format", "n5")
    if out_fmt == "n4":
        injected = np.vstack([kept_scene[:, :4], kept_obj[:, :4]])
    else:
        injected = np.vstack([kept_scene[:, :5], kept_obj[:, :5]])

    obj_mask = np.zeros(injected.shape[0], dtype=bool)
    obj_mask[kept_scene.shape[0]:] = True

    num_obj_kept = [int((kept_ids == i).sum()) for i in range(len(objects))]

    return {
        "injected": injected,
        "obj_mask": obj_mask,
        "obj_points_lidar": kept_obj,       # (M,5) surviving object points
        "obj_normals_lidar": kept_normals,   # (M,3)
        "obj_ids": kept_ids,                 # (M,) index into objects[]
        "num_scene_kept": int(kept_scene.shape[0]),
        "num_obj_kept": int(kept_obj.shape[0]),
        "num_obj_kept_per_object": num_obj_kept,
    }
