#!/usr/bin/env python
"""Regenerate the documentation figures from real pipeline runs.

The figures under ``docs/summary/figs`` are renders, not drawings, so they go
stale whenever the compositing/occlusion code changes. This rebuilds them from
two existing run directories instead of hand-editing images.

    # a single-object run (any run dir works) and a two-object run
    python scripts/make_doc_figs.py \
        --single data/outputs/p0/cmpA_pedestrian_near \
        --two    data/outputs/p0/fig_two

fig0_pipeline.png is a diagram, not a render — it is left untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CONTENT_H = 900          # panel height
LABEL_H = 64             # caption bar height
GAP = 8
CAM = "CAM_FRONT"


def _font(size=30):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # older Pillow
        return ImageFont.load_default()


def _load(path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w = int(im.width * CONTENT_H / im.height)
    return im.resize((w, CONTENT_H), Image.LANCZOS)


def _over_checkerboard(rgba_path) -> Image.Image:
    im = Image.open(rgba_path).convert("RGBA")
    w, h = int(im.width * CONTENT_H / im.height), CONTENT_H
    im = im.resize((w, h), Image.LANCZOS)
    cell = 24
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // cell) + (xx // cell)) % 2 == 0, 205, 150)
    bg = Image.fromarray(np.dstack([board] * 3).astype(np.uint8), "RGB")
    bg.paste(im, (0, 0), im)
    return bg


def _label(im: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (im.width, im.height + LABEL_H), (255, 255, 255))
    out.paste(im, (0, LABEL_H))
    d = ImageDraw.Draw(out)
    f = _font()
    bbox = d.textbbox((0, 0), text, font=f)
    d.text(((im.width - (bbox[2] - bbox[0])) // 2,
            (LABEL_H - (bbox[3] - bbox[1])) // 2 - bbox[1]), text,
           fill=(20, 20, 20), font=f)
    return out


def _grid(cells, cols=2) -> Image.Image:
    """Lay labelled panels out in a grid."""
    cells = [_label(im, t) for im, t in cells]
    cw = max(c.width for c in cells)
    ch = max(c.height for c in cells)
    rows = (len(cells) + cols - 1) // cols
    out = Image.new("RGB", (cols * cw + GAP * (cols - 1),
                            rows * ch + GAP * (rows - 1)), (255, 255, 255))
    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        out.paste(c, (col * (cw + GAP), r * (ch + GAP)))
    return out


def _half(im: Image.Image, width=780) -> Image.Image:
    h = int(im.height * width / im.width)
    return im.resize((width, h), Image.LANCZOS)


def _row(panels) -> Image.Image:
    panels = [_label(im, t) for im, t in panels]
    w = sum(p.width for p in panels) + GAP * (len(panels) - 1)
    out = Image.new("RGB", (w, panels[0].height), (255, 255, 255))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + GAP
    return out


def _frame_dir(run: Path) -> Path:
    """Resolve the directory holding one frame's artifacts.

    ``run_p0.py`` writes them straight into the run dir; ``run_batch.py`` nests
    each frame under ``sample_XXXX`` (legacy) or ``<scene>/fNNN`` (tracking).
    """
    if (run / "composite" / f"{CAM}.jpg").exists():
        return run
    hits = sorted(run.rglob(f"composite/{CAM}.jpg"))
    if not hits:
        raise SystemExit(f"no composited frame under {run}")
    return hits[0].parent.parent


def _original_image(run: Path, dataset_root: Path):
    """The unmodified camera frame a run composited onto."""
    spec_path = run / "spec.json"
    if not spec_path.exists():
        return None
    spec = json.load(open(spec_path))
    rel = spec.get("cam_filenames", {}).get(CAM)
    if not rel:
        return None
    p = dataset_root / rel
    return p if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True, help="a single-object run dir")
    ap.add_argument("--two", required=True, help="a two-object run dir")
    ap.add_argument("--dataset", default=str(ROOT / "data" / "nuscenes"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "summary" / "figs"))
    ap.add_argument("--compare", action="append", default=[],
                    metavar="LABEL=DIR",
                    help="a case for fig5, repeatable (2-4 cases read best)")
    ap.add_argument("--axis", nargs=2, metavar=("GOOD_DIR", "BAD_DIR"),
                    default=None, help="fig6: correctly vs wrongly oriented model")
    args = ap.parse_args()

    single, two = _frame_dir(Path(args.single)), _frame_dir(Path(args.two))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset)

    # ---- fig1: before / after -------------------------------------------
    orig = _original_image(single, dataset)
    panels = []
    if orig is not None:
        panels.append((_load(orig), "before - original nuScenes CAM_FRONT"))
    else:
        print(f"WARN: original frame not found for {single}, fig1 will be single-panel")
    panels.append((_load(single / "composite" / f"{CAM}.jpg"),
                   "after - injected object composited"))
    _row(panels).save(out / "fig1_before_after.jpg", quality=92)

    # ---- fig2: overlay / BEV --------------------------------------------
    panels = [(_load(single / "viz" / f"{CAM}_overlay.jpg"),
               "LiDAR points (red) + silhouette (green) on the composite")]
    bev = single / "viz" / "lidar_bev.png"
    if bev.exists():
        panels.append((_load(bev), "LiDAR BEV - target in red"))
    _row(panels).save(out / "fig2_overlay_bev.jpg", quality=92)

    # ---- fig3: two objects in one frame ---------------------------------
    _load(two / "composite" / f"{CAM}.jpg").save(out / "fig3_batch_two_objects.jpg",
                                                 quality=92)

    # ---- fig4: Blender RGBA foreground ----------------------------------
    _over_checkerboard(single / "blender" / f"{CAM}.png").save(
        out / "fig4_blender_rgba.jpg", quality=92)

    # ---- fig5: placement / category / material comparison ---------------
    if args.compare:
        cells = []
        for item in args.compare:
            label, _, run = item.partition("=")
            d = _frame_dir(Path(run))
            cells.append((_half(_load(d / "composite" / f"{CAM}.jpg")), label))
        _grid(cells, cols=2).save(out / "fig5_case_comparison.jpg", quality=92)

    # ---- fig6: axis convention (correct vs lying down) ------------------
    if args.axis:
        good, bad = (_frame_dir(Path(x)) for x in args.axis)
        cells = [(_half(_load(good / "composite" / f"{CAM}.jpg")),
                  "correct: Z-up -> Y-up before export (upright)"),
                 (_half(_load(bad / "composite" / f"{CAM}.jpg")),
                  "wrong: Z-up written into glTF (lies flat)")]
        _grid(cells, cols=2).save(out / "fig6_axis_convention.jpg", quality=92)

    for name in ("fig1_before_after.jpg", "fig2_overlay_bev.jpg",
                 "fig3_batch_two_objects.jpg", "fig4_blender_rgba.jpg",
                 "fig5_case_comparison.jpg", "fig6_axis_convention.jpg"):
        if not (out / name).exists():
            continue
        im = Image.open(out / name)
        print(f"  {out / name}  {im.size}")


if __name__ == "__main__":
    main()
