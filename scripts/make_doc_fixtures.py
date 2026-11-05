#!/usr/bin/env python
"""Regenerate every run that ``make_doc_figs.py`` needs.

The doc figures are rendered from real runs, and those runs used to be
scratch artefacts scattered outside the repo (hand-written configs in /tmp,
one throwaway script for the two-object frame). That made the figures
unreproducible: delete ``data/outputs/p0`` and the pipeline that produced them
was gone with it.

This script rebuilds all eight fixtures from definitions kept here, so a fresh
clone can refresh the figures with:

    python scripts/make_doc_fixtures.py          # -> data/outputs/p0/<fixture>
    python scripts/make_doc_figs.py ...          # command is printed at the end

Costs ~8 injections (~4 min, mostly Blender), ~200 MB under data/outputs/p0.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nusc_inject import nusc_io, pipeline  # noqa: E402
from nusc_inject.config import load_config  # noqa: E402

SAMPLE = 0
GROUND = {"search_radius_m": 4.0, "z_percentile": 15.0}

# glTF is Y-up; trimesh's exporter writes coordinates as-is, so a Z-up mesh has
# to be rotated by hand or every importer will lay the object down.
ZUP_TO_YUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
AXIS_EXTENT = (0.45, 0.55, 1.75)          # x=length, y=width, z=height
AXIS_COLOUR = (0.80, 0.80, 0.85, 1.0)


def _model(path, dim, albedo, category):
    return {"path": str(path), "target_max_dim_m": dim, "albedo": albedo,
            "intensity_gain_k": 200.0, "category": category}


def _place(offset, yaw, pitch=0.0, roll=0.0):
    return {"strategy": "ego_offset", "offset_in_ego": [offset[0], offset[1], 0.0],
            "yaw_in_ego_deg": yaw, "pitch_in_ego_deg": pitch,
            "roll_in_ego_deg": roll, "ground": dict(GROUND)}


def _one(model_path, dim, albedo, category, offset, yaw):
    """A single-object fixture: (model_cfg, placement_cfg)."""
    return {"objects": [{"model_cfg": _model(model_path, dim, albedo, category),
                         "placement_cfg": _place(offset, yaw)}]}


def build_fixtures(models: Path, assets: Path) -> dict:
    temple = assets / "temple.glb"
    return {
        # fig1 / fig2 / fig4
        "fig_temple": _one(temple, 3.0, 0.60, "movable_object.anomaly",
                           (12.0, 0.0), 30.0),
        # fig5 — A/D share a model at two ranges; B and D share a pose so the
        # only difference is the object itself
        "cmpA_pedestrian_near": _one(models / "demo_pedestrian.glb", 1.75, 0.70,
                                    "human.pedestrian.adult", (8.0, 0.0), 0.0),
        "cmpB_barrier_mid": _one(models / "demo_barrier.glb", 2.20, 0.50,
                                 "movable_object.barrier", (15.0, -5.0), 75.0),
        "cmpC_truck_far": _one(models / "demo_truck.glb", 6.00, 0.25,
                               "vehicle.truck", (28.0, 3.0), 20.0),
        "cmpD_pedestrian_mid": _one(models / "demo_pedestrian.glb", 1.75, 0.70,
                                    "human.pedestrian.adult", (15.0, -5.0), 75.0),
        # fig6 — same mesh, same pose, only the export-time axis handling differs
        "axis_ok": _one(models / "demo_axis_ok.glb", 1.75, 0.70,
                        "human.pedestrian.adult", (9.0, 1.0), 30.0),
        "axis_bad": _one(models / "demo_axis_bad.glb", 1.75, 0.70,
                         "human.pedestrian.adult", (9.0, 1.0), 30.0),
        # fig3 — two objects, different categories, explicit poses (batch's
        # random placement gave no control over the composition)
        "fig_two_final": {"objects": [
            {"model_cfg": _model(temple, 3.0, 0.60, "vehicle.car"),
             "placement_cfg": _place((13.0, -2.5), 25.0)},
            {"model_cfg": _model(models / "demo_barrier.glb", 2.20, 0.50,
                                 "movable_object.barrier"),
             "placement_cfg": _place((9.5, 2.2), 85.0)},
        ]},
    }


def write_axis_models(models: Path):
    """The same box exported correctly (Y-up) and incorrectly (Z-up raw)."""
    for name, convert in (("demo_axis_ok.glb", True), ("demo_axis_bad.glb", False)):
        mesh = trimesh.creation.box(extents=AXIS_EXTENT)
        if convert:
            mesh.vertices = np.asarray(mesh.vertices, float) @ ZUP_TO_YUP.T
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=AXIS_COLOUR, metallicFactor=0.0, roughnessFactor=0.8))
        mesh.export(models / name)
        print(f"  {models / name}  (Z-up -> Y-up: {convert})")


def main():
    models = ROOT / "data" / "models"        # 生成物：演示盒子 / 轴向对照
    assets = ROOT / "assets" / "models"      # 入库资产：temple.glb 等
    cfg = load_config(ROOT / "configs" / "base.yaml", ROOT)
    cfg["dataset"]["sample_index"] = SAMPLE
    meta = nusc_io.NuScenesMini(cfg["dataset"]["root"], cfg["dataset"]["version"])

    print("生成轴向对照模型：")
    write_axis_models(models)

    fixtures = build_fixtures(models, assets)
    out_root = ROOT / "data" / "outputs" / "p0"
    for name, spec in fixtures.items():
        out = out_root / name
        # mirror run_p0.py: same logger wiring so a fixture is inspectable the
        # same way a normal run is (logs/run.log next to the artefacts)
        (out / "logs").mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s",
                            handlers=[logging.FileHandler(out / "logs" / "run.log",
                                                          mode="w")],
                            force=True)
        res = pipeline.run_sample(cfg, meta, SAMPLE, spec["objects"], out)
        print(f"  {name:<24} status={res['status']:<8} "
              f"all_pass={res['report']['all_pass']}")
        if res["status"] != "ok":
            raise SystemExit(f"fixture {name} failed: {res['status']}")

    print("\n出图命令：")
    print("  python scripts/make_doc_figs.py \\")
    print("    --single data/outputs/p0/fig_temple \\")
    print("    --two    data/outputs/p0/fig_two_final \\")
    for key, label in (("cmpA_pedestrian_near", "A 行人 · 8 m · 居中 · 0°"),
                       ("cmpB_barrier_mid", "B 护栏 · 15 m · 左偏 5 m · 75°"),
                       ("cmpC_truck_far", "C 卡车 · 28 m · 右偏 3 m · 20°"),
                       ("cmpD_pedestrian_mid", "D 行人 · 15 m · 左偏 5 m · 75°")):
        print(f'    --compare "{label}=data/outputs/p0/{key}" \\')
    print("    --axis   data/outputs/p0/axis_ok data/outputs/p0/axis_bad")


if __name__ == "__main__":
    main()
