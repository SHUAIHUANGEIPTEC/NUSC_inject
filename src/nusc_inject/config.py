"""Config loading: YAML inheritance, plus path resolution against the repo root.

Configs express paths relative to the repository (``data/nuscenes``) so they
carry no machine-specific absolute paths. Downstream code opens them from the
process CWD (nuScenes metadata reader, trimesh, the Blender subprocess), so
they are resolved once here — every entry point then behaves identically no
matter which directory it was invoked from.

A config may declare ``extends: base.yaml`` (path relative to itself) to
inherit a shared block; the child's keys win, dicts merge recursively and
lists/scalars replace outright.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def _abs(path, root: Path) -> str:
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else (root / p).resolve())


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge of two config trees; ``override`` wins on conflict."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path, root) -> dict:
    """Load a YAML config (following ``extends``), resolving paths against ``root``."""
    root = Path(root)
    path = Path(path)
    cfg = _read(path)

    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.resolve().parent / parent_path
        cfg = _deep_merge(_read(parent_path), cfg)

    ds = cfg.setdefault("dataset", {})
    if ds.get("root"):
        ds["root"] = _abs(ds["root"], root)

    for model in cfg.get("models") or []:
        if model.get("path"):
            model["path"] = _abs(model["path"], root)
    model = cfg.get("model")
    if model and model.get("path"):
        model["path"] = _abs(model["path"], root)

    # Remember the repo root so writers can emit ROOT-RELATIVE paths into
    # deliverables (specs.json / gt.json) — a machine-specific absolute path
    # in the shipped dataset is useless on any other machine. Runtime still
    # uses the absolute form above (Blender and trimesh open by path).
    cfg["_root"] = str(root)
    return cfg
