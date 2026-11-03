#!/usr/bin/env python
"""Generate a few tiny demo models with distinct shapes and materials.

The repository ships no binary model assets, so a fresh clone has nothing to
inject. This produces three single-file `.glb` targets with clearly different
proportions and base colours, enough to exercise and compare the pipeline.

    python scripts/make_demo_models.py            # -> data/models/demo_*.glb

They are deliberately crude primitives, not substitutes for real assets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]

# name -> (extent x/y/z in metres, base colour RGBA)
#
# Axis convention follows the nuScenes box: **x = length, y = width, z = height**
# (devkit Box.corners runs `l` along box-x and `w` along box-y). A model whose
# long axis is y would still be bounded correctly but its annotation would read
# as a short, very wide object — so keep length on x.
DEMO_MODELS = {
    # standing adult: tall, roughly square footprint
    "pedestrian": ((0.45, 0.55, 1.75), (0.80, 0.80, 0.85, 1.0)),
    # road barrier: long and low
    "barrier": ((2.20, 0.40, 1.00), (0.90, 0.45, 0.10, 1.0)),
    # truck-sized body: long, wide and tall
    "truck": ((6.00, 2.40, 3.00), (0.25, 0.27, 0.30, 1.0)),
}


# glTF is Y-up by spec; trimesh's exporter writes coordinates AS-IS (it does not
# convert), so a Z-up mesh would land in the file as a spec-violating glb and
# every compliant importer would then rotate it — the object comes out lying on
# its side. Rotate to Y-up ourselves: (x, y, z) -> (x, z, -y).
ZUP_TO_YUP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float
)


def build(extent, colour):
    """A box built Z-up, written as a spec-compliant Y-up glb."""
    sx, sy, sz = extent
    mesh = trimesh.creation.box(extents=(sx, sy, sz))
    mesh.vertices = np.asarray(mesh.vertices, dtype=float) @ ZUP_TO_YUP.T
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=colour,
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "models"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (extent, colour) in DEMO_MODELS.items():
        mesh = build(extent, colour)
        path = out / f"demo_{name}.glb"
        mesh.export(path)
        print(f"{path}  extent={tuple(round(v, 2) for v in extent)}  "
              f"baseColor={tuple(round(c, 2) for c in colour[:3])}")
    print("\npoint configs at these files, e.g.")
    print("  model:\n    path: data/models/demo_barrier.glb")
    print("    target_max_dim_m: 2.2      # longest edge ≈ the real width")


if __name__ == "__main__":
    main()
