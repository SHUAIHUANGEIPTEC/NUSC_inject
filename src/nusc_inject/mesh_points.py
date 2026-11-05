"""Mesh loading + area-weighted surface sampling (host side, trimesh).

Canonical model frame = **Blender's imported frame** (so blender_render.py can
apply gt.json's T_model2global directly). glTF is Y-up by spec: trimesh keeps
the file's Y-up coordinates while Blender's importer converts to Z-up via a
-90° X rotation, so Y-up formats are converted here with the same matrix
(empirically verified against Blender 3.6.23 imported bbox).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

# Y-up -> Blender Z-up: (x, y, z) -> (x, -z, y)
YUP_TO_BLENDER = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float
)

# Formats whose files are Y-up and whose Blender importer converts to Z-up
# (glTF by spec; OBJ/FBX via the importer's axis_up='Y' default). .ply is left
# alone — it is conventionally Z-up already.
YUP_SUFFIXES = (".glb", ".gltf", ".obj", ".fbx")

# Backwards-compatible alias
GLTF_TO_BLENDER = YUP_TO_BLENDER


def _to_blender_frame(pts: np.ndarray, normals: np.ndarray, path: str):
    if Path(path).suffix.lower() in YUP_SUFFIXES:
        pts = pts @ YUP_TO_BLENDER.T
        normals = normals @ YUP_TO_BLENDER.T
    return pts, normals


def load_raw_mesh(path: str) -> trimesh.Trimesh:
    obj = trimesh.load(path, force="mesh")
    if isinstance(obj, trimesh.Scene):
        obj = obj.dump(concatenate=True)
    if not isinstance(obj, trimesh.Trimesh) or obj.faces.shape[0] == 0:
        raise ValueError(f"no triangle mesh in {path}")
    return obj


def surface_signature(verts, faces) -> np.ndarray:
    """Tessellation-independent shape summary: ``[area, cx, cy, cz]``.

    Compared between trimesh and Blender's importer as a drift guard. Two
    properties matter:

    - **surface area** is invariant to rotation and to how the loader
      tessellates the mesh (merging duplicate vertices leaves the triangles
      alone), so it verifies both sides describe the same geometry;
    - the **area-weighted centroid** is a surface integral, so it is also
      tessellation-independent — but it *moves* under an axis permutation.
      That is what a bounding box cannot see: permuting the axes of a mesh
      with a symmetric box leaves min/max untouched while the geometry is
      wrong, and the centroid is what exposes it.

    ``verts`` is (V,3) and ``faces`` is (F,3) vertex indices.
    """
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]                                   # (F, 3, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = float(areas.sum())
    if total <= 0:
        raise ValueError("mesh has zero surface area")
    centers = tri.mean(axis=1)                   # (F, 3)
    centroid = (centers * areas[:, None]).sum(axis=0) / total
    return np.concatenate([[total], centroid])


def sample_mesh(path: str, n: int):
    """Return (pts, normals, bbox_min, bbox_size, vsig, verts) in the Blender frame.

    ``pts`` is (n,3) area-weighted surface samples, ``normals`` (n,3), and the
    bbox is exact (computed from all mesh vertices, not the samples).
    ``vsig`` is ``surface_signature`` over the same mesh — see there.
    ``verts`` is (V,3) the mesh vertices themselves; ground contact must be
    computed from them, because the bbox corners are not on the mesh.

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
    return pts, normals, bmin, bsize, surface_signature(verts, mesh.faces), verts
