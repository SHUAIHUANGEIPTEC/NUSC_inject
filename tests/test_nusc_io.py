"""LiDAR .pcd.bin read/write and the keyframe-filtered metadata reader."""
from __future__ import annotations

import numpy as np
import pytest

from nusc_inject.nusc_io import load_lidar_bin, save_lidar_bin


def test_load_n5_keeps_columns(tmp_path):
    pts = np.arange(10, dtype=np.float32).reshape(2, 5)
    path = tmp_path / "a.pcd.bin"
    pts.tofile(path)
    assert np.array_equal(load_lidar_bin(path), pts)


def test_load_n4_pads_ring_column(tmp_path):
    pts = np.arange(8, dtype=np.float32).reshape(2, 4)
    path = tmp_path / "b.pcd.bin"
    pts.tofile(path)
    out = load_lidar_bin(path)
    assert out.shape == (2, 5)
    assert np.allclose(out[:, :4], pts)
    assert np.allclose(out[:, 4], 0.0)


def test_load_rejects_partial_points(tmp_path):
    path = tmp_path / "c.pcd.bin"
    np.arange(7, dtype=np.float32).tofile(path)   # neither /5 nor /4
    with pytest.raises(ValueError, match="unexpected lidar file size"):
        load_lidar_bin(path)


def test_save_creates_parent_dirs_and_roundtrips(tmp_path):
    pts = np.random.default_rng(0).normal(size=(16, 5)).astype(np.float32)
    path = tmp_path / "deep" / "nested" / "x.pcd.bin"
    save_lidar_bin(path, pts)
    assert path.exists()
    assert np.array_equal(load_lidar_bin(path), pts)


def test_save_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match=r"expected \(N,4\) or \(N,5\)"):
        save_lidar_bin(tmp_path / "x.bin", np.zeros((4, 3)))
    with pytest.raises(ValueError):
        save_lidar_bin(tmp_path / "y.bin", np.zeros(5))


def test_save_accepts_n4(tmp_path):
    pts = np.ones((3, 4), dtype=np.float64)
    path = tmp_path / "n4.bin"
    save_lidar_bin(path, pts)
    out = load_lidar_bin(path)
    assert out.shape == (3, 5)        # reload pads ring
    assert np.allclose(out[:, :4], 1.0)
