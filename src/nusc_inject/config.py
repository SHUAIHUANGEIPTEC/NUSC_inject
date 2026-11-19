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

import csv
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


def _expand_models_from(cfg: dict, root: Path) -> None:
    """Turn a ``models_from`` asset-tag filter into extra ``models`` entries.

    The road-OOD pool spans thousands of flat files, so a hand-written YAML
    list is not an option. ``road_ood_asset_tags.csv`` (written by
    ``scripts/classify_id_ood.py``) is the single source of truth for
    ID/OOD tiering; this turns a column filter (e.g. ``tier: OOD``) into
    pool entries shaped exactly like static ``models:`` items, so the batch
    planner needs no changes. Explicit ``models:`` entries are kept and
    come first.

    Keys of ``models_from``:
      tags_csv       asset-tags CSV path (repo-relative)
      filter         {column: value} — a row matches when ALL pairs match
      defaults       fields merged into every row (albedo / category / ...)
      size_by_class  {lvis_class: [min_m, max_m]} per-class realistic size —
                     sanity gate: a fire extinguisher must not scale to 3 m
                     just because its group allows it (beats size_by_group)
      size_by_group  {group: [min_m, max_m]} per-group random size range
      default_size   [min_m, max_m] fallback when neither table lists it
      base_dir       directory the ``file`` column is relative to
                     (repo-relative; default ``assets/models``)
    """
    spec = cfg.get("models_from")
    if not spec:
        return
    tags_csv = _abs(spec["tags_csv"], root)
    filters = dict(spec.get("filter") or {})
    defaults = dict(spec.get("defaults") or {})
    size_by_class = dict(spec.get("size_by_class") or {})
    size_by_group = dict(spec.get("size_by_group") or {})
    default_size = list(spec.get("default_size") or [0.5, 3.0])
    base_dir = _abs(spec.get("base_dir") or "assets/models", root)

    pool: list[dict] = []
    with open(tags_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if any(str(row.get(k, "")) != str(v) for k, v in filters.items()):
                continue
            fname = row.get("file") or ""
            if not fname:
                continue
            file_path = Path(base_dir) / fname
            if not file_path.is_file():
                continue
            entry = dict(defaults)
            entry["path"] = str(file_path.resolve())
            # realistic per-class size wins; group table and default follow
            entry["target_max_dim_m"] = list(
                size_by_class.get(row.get("lvis_class") or "")
                or size_by_group.get(row.get("group") or "", default_size))
            pool.append(entry)

    if not pool:
        raise ValueError(
            f"models_from matched 0 assets: csv={tags_csv} filter={filters}")
    cfg["models"] = list(cfg.get("models") or []) + pool


def load_config(path, root) -> dict:
    """Load a YAML config (following ``extends``), resolving paths against ``root``."""
    root = Path(root)
    path = Path(path)
    cfg = _read(path)

    # Flatten a chain of ``extends`` (child -> parent -> ... -> base). The
    # parent file may itself declare ``extends``; resolving it in a loop
    # keeps the merge one pass and lets each hop resolve its own relative
    # parent against ITS directory (configs/child -> configs/base).
    while True:
        parent = cfg.pop("extends", None)
        if not parent:
            break
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.resolve().parent / parent_path
        cfg = _deep_merge(_read(parent_path), cfg)
        path = parent_path

    ds = cfg.setdefault("dataset", {})
    if ds.get("root"):
        ds["root"] = _abs(ds["root"], root)

    for model in cfg.get("models") or []:
        if model.get("path"):
            model["path"] = _abs(model["path"], root)
    model = cfg.get("model")
    if model and model.get("path"):
        model["path"] = _abs(model["path"], root)

    # asset-tags filter -> more ``models`` entries (after static ones so an
    # explicit hand-picked model always stays at the head of the pool)
    _expand_models_from(cfg, root)

    # Remember the repo root so writers can emit ROOT-RELATIVE paths into
    # deliverables (specs.json / gt.json) — a machine-specific absolute path
    # in the shipped dataset is useless on any other machine. Runtime still
    # uses the absolute form above (Blender and trimesh open by path).
    cfg["_root"] = str(root)
    return cfg
