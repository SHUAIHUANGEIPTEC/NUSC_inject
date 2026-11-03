#!/usr/bin/env python
"""Regenerate the documentation figures from real pipeline runs.

The figures under ``docs/summary/figs`` are renders, not drawings, so they go
stale whenever the compositing/occlusion code changes. This rebuilds them from
existing run directories instead of hand-editing images.

Injected objects occupy only ~0.6-4 % of a 1600x900 frame, so a plain
full-frame crop is unreadable; every panel therefore carries a highlighted ROI
box plus an enlarged inset of that box.

    python scripts/make_doc_figs.py \
        --single data/outputs/p0/fig_temple \
        --two    data/outputs/p0/fig_two_final \
        --compare "A 行人 8m=..." --compare "..." \
        --axis   data/outputs/p0/axis_ok data/outputs/p0/axis_bad

fig0_pipeline.png is a diagram, not a render — it is left untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CONTENT_H = 900           # native panel height before downscaling
LABEL_H = 62
GAP = 10
CAM = "CAM_FRONT"
PANEL_W = 780             # panel width in the composed figure
INSET_FRAC = 0.44         # inset width as a fraction of the panel

CJK_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)


def _font(size=28):
    for p in CJK_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size), True
            except OSError:
                continue
    return ImageFont.load_default(), False


FONT, HAS_CJK = _font()

# Labels are Chinese to match the docs; fall back to ASCII when no CJK font
# is installed so the text never renders as boxes.
T = {
    "before": "注入前 · 原始 CAM_FRONT" if HAS_CJK else "before - original CAM_FRONT",
    "after": "注入后 · 目标已合成" if HAS_CJK else "after - composite",
    "overlay": "注入的 LiDAR 点（红）落在渲染轮廓（绿）内" if HAS_CJK
               else "LiDAR points (red) inside the silhouette (green)",
    "bev": "BEV · 目标为红色" if HAS_CJK else "BEV - target in red",
    "rgba": "Blender RGBA 前景（棋盘格 = 透明）" if HAS_CJK
            else "Blender RGBA foreground (checker = transparent)",
    "zoom": "放大" if HAS_CJK else "zoom",
}


# ------------------------------------------------------------------ primitives


def _load(path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im.resize((int(im.width * CONTENT_H / im.height), CONTENT_H), Image.LANCZOS)


def _half(im: Image.Image, width=PANEL_W) -> Image.Image:
    h = int(im.height * width / im.width)
    return im.resize((width, h), Image.LANCZOS)


def _label(im: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (im.width, im.height + LABEL_H), (255, 255, 255))
    out.paste(im, (0, LABEL_H))
    d = ImageDraw.Draw(out)
    bbox = d.textbbox((0, 0), text, font=FONT)
    d.text(((im.width - (bbox[2] - bbox[0])) // 2,
            (LABEL_H - (bbox[3] - bbox[1])) // 2 - bbox[1]), text,
           fill=(25, 25, 25), font=FONT)
    return out


def _row(panels) -> Image.Image:
    panels = [_label(im, t) for im, t in panels]
    w = sum(p.width for p in panels) + GAP * (len(panels) - 1)
    out = Image.new("RGB", (w, max(p.height for p in panels)), (255, 255, 255))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + GAP
    return out


def _grid(cells, cols=2) -> Image.Image:
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


# ------------------------------------------------------------------ zoom inset


def _roi_from_alpha(png_path, pad_frac=0.35):
    """Bounding box of the rendered silhouette, padded, in native pixels."""
    arr = np.array(Image.open(png_path).convert("RGBA"))
    rows, cols = np.where(arr[:, :, 3] > 5)
    if len(rows) == 0:
        return None
    h, w = arr.shape[:2]
    x0, x1, y0, y1 = cols.min(), cols.max(), rows.min(), rows.max()
    pad = int(max(x1 - x0, y1 - y0) * pad_frac) + 12
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(w, x1 + pad), min(h, y1 + pad))


def _with_zoom(im: Image.Image, roi, box_rgb=(255, 200, 0),
               native_w: int = 1600) -> Image.Image:
    """Annotate ``im`` with the ROI rectangle and paste an enlarged inset.

    ``roi`` is in native frame pixels; the inset goes to the corner farthest
    from the ROI so it never covers it.
    """
    if roi is None:
        return im
    out = im.copy()
    W = out.width
    scale = W / float(native_w)
    x0, y0, x1, y1 = (int(v * scale) for v in roi)
    pad = 4
    ImageDraw.Draw(out).rectangle(
        [x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=box_rgb, width=3)

    crop = out.crop((x0, y0, x1, y1))
    iw = int(W * INSET_FRAC)
    ih = max(1, int(crop.height * iw / max(crop.width, 1)))
    inset = crop.resize((iw, ih), Image.LANCZOS)
    b = 4
    framed = Image.new("RGB", (iw + 2 * b, ih + 2 * b), box_rgb)
    framed.paste(inset, (b, b))

    right = (x0 + x1) / 2 < W / 2
    bottom = (y0 + y1) / 2 < out.height / 2
    px = W - framed.width - 14 if right else 14
    py = out.height - framed.height - 14 if bottom else 14
    out.paste(framed, (px, py))
    return out


def _object_rois(gt_path, job_path):
    """Screen-space bbox per object, by projecting its raw bbox corners."""
    gts = json.load(open(gt_path))
    gts = gts if isinstance(gts, list) else [gts]
    job = json.load(open(job_path))
    cam = next((c for c in job["cams"] if c["name"] == CAM), None)
    if cam is None:
        return []
    K = np.array(cam["K"], dtype=float)
    T_g2c = np.linalg.inv(np.array(cam["T_cam2global"], dtype=float).reshape(4, 4))
    out = []
    for gt in gts:
        T = np.array(gt["T_model2global"], dtype=float).reshape(4, 4)
        lo = np.array(gt["raw_bbox_min"], dtype=float)
        sz = np.array(gt["raw_bbox_size"], dtype=float)
        # note: bx/by/bz, NOT sx/sy/sz — the latter would shadow the size vector
        corners = np.array([lo + np.array([bx, by, bz]) * sz
                            for bx in (0, 1) for by in (0, 1) for bz in (0, 1)])
        hom = np.hstack([corners, np.ones((8, 1))])
        pc = (T_g2c @ (T @ hom.T)).T[:, :3]
        front = pc[:, 2] > 0.1
        if not front.any():
            continue
        proj = pc[front] @ K.T
        pix = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)
        pad = 30
        out.append((max(0, pix[:, 0].min() - pad), max(0, pix[:, 1].min() - pad),
                    pix[:, 0].max() + pad, pix[:, 1].max() + pad))
    return out


# ---------------------------------------------------------------- run plumbing


def _frame_dir(run: Path) -> Path:
    """Resolve the directory holding one frame's artifacts."""
    if (run / "composite" / f"{CAM}.jpg").exists():
        return run
    hits = sorted(run.rglob(f"composite/{CAM}.jpg"))
    if not hits:
        raise SystemExit(f"no composited frame under {run}")
    return hits[0].parent.parent


def _original_image(run: Path, dataset_root: Path):
    spec_path = run / "spec.json"
    if not spec_path.exists():
        return None
    rel = json.load(open(spec_path)).get("cam_filenames", {}).get(CAM)
    if not rel:
        return None
    p = dataset_root / rel
    return p if p.exists() else None


def _bev_crop(run: Path, radius_m=16.0):
    """Crop the 800x800 BEV (range 40 m, res 0.1) around the injected object."""
    bev = run / "viz" / "lidar_bev.png"
    gt = run / "gt.json"
    if not bev.exists() or not gt.exists():
        return None
    g = json.load(open(gt))
    g = g[0] if isinstance(g, list) else g
    xy = g.get("placement", {}).get("offset_in_lidar_frame")
    if not xy:
        return None
    im = Image.open(bev).convert("RGB")
    rng_m, res = 40.0, 0.1
    u = int((xy[0] + rng_m) / res)
    v = int((rng_m - xy[1]) / res)
    r = int(radius_m / res)
    box = (max(0, u - r), max(0, v - r),
           min(im.width, u + r), min(im.height, v + r))
    return im.crop(box)


# ----------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True, help="a single-object run dir")
    ap.add_argument("--two", required=True, help="a two-object run dir")
    ap.add_argument("--dataset", default=str(ROOT / "data" / "nuscenes"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "summary" / "figs"))
    ap.add_argument("--compare", action="append", default=[], metavar="LABEL=DIR")
    ap.add_argument("--axis", nargs=2, metavar=("GOOD_DIR", "BAD_DIR"), default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset)
    single = _frame_dir(Path(args.single))

    # ---- fig1: before / after, both zoomed on the injected region --------
    roi = _roi_from_alpha(single / "blender" / f"{CAM}.png")
    panels = []
    orig = _original_image(single, dataset)
    if orig is not None:
        panels.append((_with_zoom(_half(_load(orig)), roi), T["before"]))
    else:
        print(f"WARN: original frame not found for {single}")
    panels.append((_with_zoom(_half(_load(single / "composite" / f"{CAM}.jpg")), roi),
                   T["after"]))
    _row(panels).save(out / "fig1_before_after.jpg", quality=92)

    # ---- fig2: overlay (zoomed) + BEV cropped around the target ----------
    panels = [(_with_zoom(_half(_load(single / "viz" / f"{CAM}_overlay.jpg")), roi),
               T["overlay"])]
    bev = _bev_crop(single)
    if bev is not None:
        # match the camera panel's content height so the row has no dead space
        side = int(PANEL_W * CONTENT_H / 1600)
        panels.append((bev.resize((side, side), Image.LANCZOS), T["bev"]))
    _row(panels).save(out / "fig2_overlay_bev.jpg", quality=92)

    # ---- fig3: two objects, one ROI box + inset each ---------------------
    two = _frame_dir(Path(args.two))
    img = _half(_load(two / "composite" / f"{CAM}.jpg"))
    rois = _object_rois(two / "gt.json", two / "job.json")
    if not rois:
        rois = [_roi_from_alpha(two / "blender" / f"{CAM}.png")]
    for r in rois:
        img = _with_zoom(img, r)
    text = ("同帧 2 目标 · 不同类目（黄框已放大）" if HAS_CJK
            else "two objects, per-object category (yellow ROI zoomed)")
    _label(img, text).save(out / "fig3_batch_two_objects.jpg", quality=92)

    # ---- fig4: Blender RGBA on a checkerboard ----------------------------
    arr = np.array(Image.open(single / "blender" / f"{CAM}.png").convert("RGBA"))
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // 24) + (xx // 24)) % 2 == 0, 205, 150)
    bg = Image.fromarray(np.dstack([board] * 3).astype(np.uint8), "RGB")
    bg.paste(Image.fromarray(arr, "RGBA"), (0, 0), Image.fromarray(arr, "RGBA"))
    _label(_with_zoom(_half(bg), roi), T["rgba"]).save(
        out / "fig4_blender_rgba.jpg", quality=92)

    # ---- fig5: placement / category / material comparison ----------------
    if args.compare:
        cells = []
        for item in args.compare:
            label, _, run = item.partition("=")
            d = _frame_dir(Path(run))
            im = _half(_load(d / "composite" / f"{CAM}.jpg"))
            cells.append((_with_zoom(im, _roi_from_alpha(d / "blender" / f"{CAM}.png")),
                          label))
        _grid(cells, cols=2).save(out / "fig5_case_comparison.jpg", quality=92)

    # ---- fig6: axis convention (upright vs lying flat) -------------------
    if args.axis:
        cells = []
        texts = (("正确：导出前 Z-up→Y-up（站立）", "错误：Z-up 直接写进 glTF（躺平）")
                 if HAS_CJK else
                 ("correct: Z-up -> Y-up before export (upright)",
                  "wrong: Z-up written into glTF (lies flat)"))
        for path, text in zip(args.axis, texts):
            d = _frame_dir(Path(path))
            im = _half(_load(d / "composite" / f"{CAM}.jpg"))
            cells.append((_with_zoom(im, _roi_from_alpha(d / "blender" / f"{CAM}.png")),
                          text))
        _grid(cells, cols=2).save(out / "fig6_axis_convention.jpg", quality=92)

    for name in ("fig1_before_after.jpg", "fig2_overlay_bev.jpg",
                 "fig3_batch_two_objects.jpg", "fig4_blender_rgba.jpg",
                 "fig5_case_comparison.jpg", "fig6_axis_convention.jpg"):
        if (out / name).exists():
            print(f"  {out / name}  {Image.open(out / name).size}")


if __name__ == "__main__":
    main()
