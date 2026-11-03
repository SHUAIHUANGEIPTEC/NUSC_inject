"""Config loading: ``extends`` merging and repo-relative path resolution."""
from __future__ import annotations

from pathlib import Path

import pytest

from nusc_inject.config import load_config

BASE = """
dataset:
  root: data/nuscenes
  version: v1.0-mini
  cam_channels: [CAM_FRONT, CAM_BACK]
composite:
  alpha_threshold: 0.05
  depth_margin_m: 0.5
validation: &val
  min_coverage: 0.85
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / "configs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_extends_merges_dicts_and_child_wins(tmp_path):
    _write(tmp_path, "base.yaml", BASE)
    child = _write(tmp_path, "child.yaml", """
extends: base.yaml
dataset:
  version: v1.0-trainval
composite:
  depth_margin_m: 1.5
""")
    cfg = load_config(child, tmp_path)

    assert "extends" not in cfg
    # inherited keys survive the merge
    assert cfg["dataset"]["cam_channels"] == ["CAM_FRONT", "CAM_BACK"]
    assert cfg["validation"]["min_coverage"] == 0.85
    # child overrides win, sibling keys under the same parent are kept
    assert cfg["dataset"]["version"] == "v1.0-trainval"
    assert cfg["composite"]["depth_margin_m"] == 1.5
    assert cfg["composite"]["alpha_threshold"] == 0.05


def test_lists_replace_instead_of_concatenating(tmp_path):
    _write(tmp_path, "base.yaml", BASE)
    child = _write(tmp_path, "child.yaml", """
extends: base.yaml
dataset:
  cam_channels: [CAM_FRONT]
""")
    cfg = load_config(child, tmp_path)
    assert cfg["dataset"]["cam_channels"] == ["CAM_FRONT"]


def test_relative_paths_resolve_against_root(tmp_path):
    _write(tmp_path, "base.yaml", BASE)
    child = _write(tmp_path, "child.yaml", """
extends: base.yaml
model:
  path: data/models/a.glb
models:
  - path: data/models/b.glb
  - path: /abs/already/c.glb
""")
    cfg = load_config(child, tmp_path)

    assert cfg["dataset"]["root"] == str((tmp_path / "data/nuscenes").resolve())
    assert cfg["model"]["path"] == str((tmp_path / "data/models/a.glb").resolve())
    assert cfg["models"][0]["path"] == str((tmp_path / "data/models/b.glb").resolve())
    assert cfg["models"][1]["path"] == "/abs/already/c.glb"


def test_missing_parent_raises(tmp_path):
    child = _write(tmp_path, "child.yaml", "extends: nope.yaml\n")
    with pytest.raises(FileNotFoundError):
        load_config(child, tmp_path)


def test_repo_configs_load_and_keep_their_own_keys():
    """The shipped configs must merge cleanly and stay within their role."""
    root = Path(__file__).resolve().parents[1]
    p0 = load_config(root / "configs" / "p0.yaml", root)
    batch = load_config(root / "configs" / "batch.yaml", root)

    for cfg in (p0, batch):
        assert cfg["dataset"]["version"] == "v1.0-mini"
        assert cfg["blender"]["binary"] == "tools/blender/blender"
        assert Path(cfg["dataset"]["root"]).is_absolute()

    assert "sample_index" in p0["dataset"] and "model" in p0
    assert "sample_indices" in batch["dataset"] and "models" in batch
