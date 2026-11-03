"""Mesh sampling and the cross-implementation drift signature."""
from __future__ import annotations

import numpy as np
import pytest

from nusc_inject import mesh_points as mp


# --------------------------------------------------------- surface_signature


def _cube(center=(0.0, 0.0, 0.0), size=2.0):
    s = size / 2.0
    v = np.array([[sx * s, sy * s, sz * s] for sx in (-1, 1)
                  for sy in (-1, 1) for sz in (-1, 1)]) + np.asarray(center)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def test_surface_signature_is_stable_for_identical_meshes():
    v, f = _cube()
    a = mp.surface_signature(v, f)
    assert a[0] == pytest.approx(6 * 2.0 ** 2)      # 6 faces of 2x2
    assert np.allclose(a[1:], [0.0, 0.0, 0.0])      # centred cube
    assert np.allclose(a, mp.surface_signature(v.copy(), f.copy()))


def test_surface_signature_is_tessellation_independent():
    """Splitting triangles must not change the summary — only the shape may."""
    v, f = _cube()
    base = mp.surface_signature(v, f)

    # subdivide every triangle into two by duplicating its centroid vertex
    v2 = list(v)
    f2 = []
    for a, b, c in f:
        mid = (v[a] + v[b] + v[c]) / 3.0
        k = len(v2)
        v2.append(mid)
        f2 += [[a, b, k], [b, c, k], [c, a, k]]
    split = mp.surface_signature(np.array(v2), np.array(f2))
    assert np.allclose(split, base, rtol=1e-9)


def test_surface_signature_moves_the_centroid_with_the_geometry():
    v, f = _cube(center=(1.0, 2.0, 3.0))
    s = mp.surface_signature(v, f)
    assert np.allclose(s[1:], [1.0, 2.0, 3.0])


def test_surface_signature_detects_an_axis_mix_up_the_bbox_cannot_see():
    """A cube-shaped bbox is invariant under axis permutation; the centroid is not.

    This is the blind spot the drift guard used to have: swapping Y and Z of a
    mesh whose bounding box is symmetric in y and z leaves min/max untouched
    while the geometry is wrong.
    """
    v = np.array([[ 1.0,  1.0,  1.0],
                  [ 1.0, -1.0, -1.0],
                  [-1.0,  1.0, -1.0],
                  [-1.0, -0.6,  0.6]])
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    swapped = v[:, [0, 2, 1]]

    # the bbox really is blind to this swap ...
    assert np.allclose(v.min(axis=0), swapped.min(axis=0))
    assert np.allclose(v.max(axis=0), swapped.max(axis=0))
    # ... while the surface centroid is not
    a = mp.surface_signature(v, f)
    b = mp.surface_signature(swapped, f)
    assert a[0] == pytest.approx(b[0])              # area is rotation-invariant
    assert not np.allclose(a[1:], b[1:]), "centroid must flag the permutation"


def test_surface_signature_rejects_a_degenerate_mesh():
    v, f = _cube()
    with pytest.raises(ValueError, match="zero surface area"):
        mp.surface_signature(np.zeros_like(v), f)


# --------------------------------------------------------- axis conversion


def test_yup_formats_are_converted_to_blender_frame():
    pts = np.array([[1.0, 2.0, 3.0]])
    normals = np.array([[0.0, 1.0, 0.0]])
    for suffix in (".glb", ".gltf", ".obj", ".fbx"):
        out, n = mp._to_blender_frame(pts, normals, f"model{suffix}")
        assert np.allclose(out, [[1.0, -3.0, 2.0]]), suffix
        assert np.allclose(n, [[0.0, 0.0, 1.0]]), suffix


def test_ply_is_left_in_the_file_frame():
    pts = np.array([[1.0, 2.0, 3.0]])
    out, _ = mp._to_blender_frame(pts, np.zeros((1, 3)), "model.ply")
    assert np.allclose(out, [[1.0, 2.0, 3.0]])


def test_yup_suffix_set_is_documented():
    assert mp.YUP_SUFFIXES == (".glb", ".gltf", ".obj", ".fbx")


# ---------------------------------------------------------- load_raw_mesh


def test_load_raw_mesh_rejects_a_non_mesh(tmp_path):
    p = tmp_path / "empty.obj"
    p.write_text("# no geometry\n")
    with pytest.raises(ValueError, match="no triangle mesh"):
        mp.load_raw_mesh(str(p))
