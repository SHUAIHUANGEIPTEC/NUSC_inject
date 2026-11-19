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


def test_extends_chain_flattens(tmp_path):
    """child -> mid -> base: a two-hop chain must fully merge."""
    _write(tmp_path, "base.yaml", BASE)
    _write(tmp_path, "mid.yaml", """
extends: base.yaml
composite:
  depth_margin_m: 2.0
""")
    child = _write(tmp_path, "child.yaml", """
extends: mid.yaml
dataset:
  version: v1.0-trainval
""")
    cfg = load_config(child, tmp_path)

    assert "extends" not in cfg
    # from base, two hops up
    assert cfg["dataset"]["cam_channels"] == ["CAM_FRONT", "CAM_BACK"]
    # from mid
    assert cfg["composite"]["depth_margin_m"] == 2.0
    # child wins over mid
    assert cfg["dataset"]["version"] == "v1.0-trainval"


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


# ------------------------------------------------------------- models_from

TAGS_HEADER = "file,lvis_class,group,tier,is_ood,ood_category\n"


def _write_pool(tmp_path: Path) -> Path:
    """Asset dir with two .glb files + an asset-tags CSV covering both."""
    models = tmp_path / "assets" / "models"
    models.mkdir(parents=True)
    (models / "cat_ood.glb").write_bytes(b"glTF")
    (models / "cone_id.glb").write_bytes(b"glTF")
    tags = tmp_path / "tags.csv"
    tags.write_text(
        TAGS_HEADER
        + "cat_ood.glb,cat,animals,OOD,1,movable_object.anomaly\n"
        + "cone_id.glb,cone,road_facilities,ID-CORE,0,\n"
        + "missing.glb,ghost,animals,OOD,1,movable_object.anomaly\n",
        encoding="utf-8")
    return tags


def _models_from_cfg() -> str:
    return """
models_from:
  tags_csv: tags.csv
  filter: { tier: OOD }
  defaults:
    albedo: 0.6
    intensity_gain_k: 200.0
    category: movable_object.anomaly
  size_by_group:
    animals: [0.4, 2.0]
  default_size: [0.5, 3.0]
"""


def test_models_from_filters_and_shapes_pool(tmp_path):
    tags = _write_pool(tmp_path)
    child = _write(tmp_path, "child.yaml", _models_from_cfg().replace(
        "tags.csv", str(tags.relative_to(tmp_path))))
    cfg = load_config(child, tmp_path)

    # only the tier=OOD row matches; the missing file row is skipped
    assert len(cfg["models"]) == 1
    m = cfg["models"][0]
    assert m["path"] == str((tmp_path / "assets/models/cat_ood.glb").resolve())
    assert m["albedo"] == 0.6
    assert m["category"] == "movable_object.anomaly"
    # per-group size range applied
    assert m["target_max_dim_m"] == [0.4, 2.0]


def test_models_from_keeps_static_models_first(tmp_path):
    tags = _write_pool(tmp_path)
    child = _write(tmp_path, "child.yaml", """
models:
  - path: data/models/hand.glb
""" + _models_from_cfg().replace("tags.csv", str(tags.relative_to(tmp_path))))
    (tmp_path / "data" / "models").mkdir(parents=True)
    (tmp_path / "data" / "models" / "hand.glb").write_bytes(b"glTF")

    cfg = load_config(child, tmp_path)
    assert len(cfg["models"]) == 2
    assert cfg["models"][0]["path"].endswith("hand.glb")
    assert cfg["models"][1]["path"].endswith("cat_ood.glb")


def test_size_by_class_beats_group(tmp_path):
    """A class-level realistic size wins over the (coarser) group range."""
    tags = _write_pool(tmp_path)
    child = _write(tmp_path, "child.yaml", _models_from_cfg().replace(
        "  size_by_group:",
        "  size_by_class:\n    cat: [0.3, 0.6]\n  size_by_group:").replace(
        "tags.csv", str(tags.relative_to(tmp_path))))
    cfg = load_config(child, tmp_path)

    assert cfg["models"][0]["target_max_dim_m"] == [0.3, 0.6]


def test_models_from_zero_matches_raises(tmp_path):
    tags = _write_pool(tmp_path)
    child = _write(tmp_path, "child.yaml", _models_from_cfg().replace(
        "tier: OOD", "tier: NO_SUCH_TIER").replace(
        "tags.csv", str(tags.relative_to(tmp_path))))
    with pytest.raises(ValueError, match="matched 0 assets"):
        load_config(child, tmp_path)


def test_models_from_absent_is_noop(tmp_path):
    _write(tmp_path, "child.yaml", "dataset:\n  version: v1.0-mini\n")
    cfg = load_config(tmp_path / "configs" / "child.yaml", tmp_path)
    assert "models" not in cfg or not cfg.get("models")


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
