"""Mesh loading + area-weighted surface sampling (host side, trimesh).

Canonical model frame = **Blender's imported frame** (so blender_render.py can
apply gt.json's T_model2global directly). glTF is Y-up by spec: trimesh keeps
the file's Y-up coordinates while Blender's importer converts to Z-up via a
-90° X rotation, so glb/gltf samples are converted here with the same matrix
(empirically verified against Blender 3.6.23 imported bbox).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

# glTF Y-up -> Blender Z-up: (x, y, z) -> (x, -z, y)
GLTF_TO_BLENDER = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float
)


def _to_blender_frame(pts: np.ndarray, normals: np.ndarray, path: str):
    if Path(path).suffix.lower() in (".glb", ".gltf"):
        pts = pts @ GLTF_TO_BLENDER.T
        normals = normals @ GLTF_TO_BLENDER.T
    return pts, normals


def load_raw_mesh(path: str) -> trimesh.Trimesh:
    obj = trimesh.load(path, force="mesh")
    if isinstance(obj, trimesh.Scene):
        obj = obj.dump(concatenate=True)
    if not isinstance(obj, trimesh.Trimesh) or obj.faces.shape[0] == 0:
        raise ValueError(f"no triangle mesh in {path}")
    return obj


def sample_mesh(path: str, n: int):
    """Return (pts (n,3), normals (n,3), bbox_min, bbox_size) in Blender frame.

    Area-weighted surface sampling (trimesh.sample.sample_surface); the bbox
    is exact (computed from all mesh vertices, not the samples).

    ``seed`` is pinned: trimesh samples from OS entropy by default, which made
    every run inject a different point set (and therefore a different
    num_lidar_pts / validation result) for the same config.
    """
    mesh = load_raw_mesh(path)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n, seed=0)
    normals = mesh.face_normals[face_idx]
    pts, normals = _to_blender_frame(
        np.asarray(pts, dtype=float), np.asarray(normals, dtype=float), path
    )
    verts, _ = _to_blender_frame(
        np.asarray(mesh.vertices, dtype=float),
        np.zeros((len(mesh.vertices), 3)), path,
    )
    bmin = verts.min(axis=0)
    bsize = verts.max(axis=0) - bmin
    return pts, normals, bmin, bsize
