#!/usr/bin/env python3
"""渲染 OOD 注入序列动图：单相机连续视场 + 2D bbox 圈注 + 距离约束。

扫描 `data/outputs/OoD/<run_id>/` 的逐帧 spec（`run_batch.py` 产物），自动选出
「同一相机上帧号连续、且注入目标距该相机 < `--max-dist` 米」的最优轨迹；按
nuScenes 官方标定把注入目标的 3D 框（T_model2global 内嵌 scale，raw bbox 角点
直接左乘）投影为 2D AABB，画黄色框 + 类目/距离标签，PIL 合成循环 GIF。

用法:
    python scripts/make_ood_gif.py                       # 自动选轨 -> data/outputs/OoD/ood_gif/
    python scripts/make_ood_gif.py --cam CAM_FRONT --max-dist 20 --fps 2
    python scripts/make_ood_gif.py --run odd_smoke       # 指定运行目录
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from nusc_inject import camera_channel, nusc_io, transforms as tf
from nusc_inject.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def load_sidecar_cn() -> dict[str, str]:
    """{glb 文件名: 中文类目}，来自 road_ood sidecar（缺失时回退类目前缀）。"""
    cn = {}
    for side in (ROOT / "assets" / "models").glob("*.json"):
        try:
            d = json.loads(side.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if d.get("lvis_class"):
            cn[f"{side.stem}.glb"] = d.get("class_cn") or d["lvis_class"]
    return cn


def scan_run(run_dir: Path):
    """逐帧 spec -> 按 track 分组（保持帧序），附各帧已有 composite 相机。"""
    frames = []
    for spec_path in sorted(run_dir.glob("scene-*/f*/spec.json")):
        d = spec_path.parent
        spec = json.loads(spec_path.read_text())
        cams = sorted(p.stem for p in (d / "composite").glob("*.jpg"))
        fidx = int(re.search(r"f(\d+)$", d.name).group(1))
        frames.append({"fidx": fidx, "dir": d, "tok": spec["sample_token"],
                       "objects": spec["objects"], "cams": cams})
    by_track = defaultdict(list)
    for fr in sorted(frames, key=lambda x: x["fidx"]):
        for i, o in enumerate(fr["objects"]):
            tk = o.get("track_key") or f"single_{fr['fidx']}_{i}"
            by_track[tk].append({**fr, "oi": i})
    return by_track


def frame_gts(frame) -> list[dict]:
    g = json.loads((frame["dir"] / "gt.json").read_text())
    return g if isinstance(g, list) else [g]


def project_box_AABB(gt: dict, cam_spec: dict) -> tuple[np.ndarray, float]:
    """3D 框(T 内嵌 scale) -> 2D AABB；返回 (xyxy, 前向角点最小 z)。"""
    T = np.asarray(gt["T_model2global"], dtype=float).reshape(4, 4)
    rmin = np.asarray(gt["raw_bbox_min"], dtype=float)
    rsz = np.asarray(gt["raw_bbox_size"], dtype=float)
    corners = np.array([rmin + np.array([a, b, c]) * rsz
                        for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    hom = np.hstack([corners, np.ones((8, 1))])
    pts_g = (T @ hom.T).T[:, :3]
    K = np.asarray(cam_spec["K"], dtype=float)
    T_g2c = np.linalg.inv(np.asarray(cam_spec["T_cam2global"], dtype=float).reshape(4, 4))
    pts_c = tf.transform_pts(T_g2c, pts_g)
    front = pts_c[:, 2] > 0.1
    if not front.any():
        return None, float("inf")
    pix = tf.view_points(pts_c[front], K)
    zmin = float(pts_c[front, 2].min())
    h, w = cam_spec["height"], cam_spec["width"]
    xyxy = np.array([max(0, pix[:, 0].min()), max(0, pix[:, 1].min()),
                     min(w - 1, pix[:, 0].max()), min(h - 1, pix[:, 1].max())])
    return xyxy, zmin


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="ood_nuscenes_v1", help="data/outputs/OoD/ 下的运行目录名")
    ap.add_argument("--cam", default="CAM_FRONT")
    ap.add_argument("--max-dist", type=float, default=20.0, help="目标距相机上限（米）")
    ap.add_argument("--min-frames", type=int, default=3, help="连续帧数下限")
    ap.add_argument("--fps", type=float, default=2.0, help="GIF 帧率（2 = nuScenes 原速 0.5s/帧）")
    ap.add_argument("--width", type=int, default=1280, help="输出宽度（高度按比例）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "ood_nuscenes.yaml", ROOT)
    meta = nusc_io.NuScenesMini(cfg["dataset"]["root"], cfg["dataset"]["version"])
    run_dir = ROOT / "data" / "outputs" / "OoD" / args.run
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    cn_map = load_sidecar_cn()
    by_track = scan_run(run_dir)

    # ---- 选轨：帧号连续 + 全程 < max_dist，帧多优先、距离近优先 ----
    candidates = []
    for tk, frs in by_track.items():
        have = [f for f in frs if args.cam in f["cams"]]
        if len(have) < args.min_frames:
            continue
        fidxs = [f["fidx"] for f in have]
        if any(b - a != 1 for a, b in zip(fidxs, fidxs[1:])):
            continue
        dists, valid = [], True
        for f in have:
            gts = frame_gts(f)
            if f["oi"] >= len(gts):
                valid = False; break
            g = gts[f["oi"]]
            if Path(g["model_path"]).name != Path(f["objects"][f["oi"]]["model_path"]).name:
                valid = False; break
            sd = meta.sd_for_sample(f["tok"], args.cam)
            T_g2c = camera_channel.global2cam(sd, meta)
            cam_t = np.linalg.inv(T_g2c)[:3, 3]
            t = np.asarray(g["T_model2global"], float).reshape(4, 4)[:3, 3]
            dists.append(float(np.linalg.norm(t - cam_t)))
        if valid and dists and max(dists) <= args.max_dist:
            candidates.append((len(have), -min(dists), tk, have, dists))
    if not candidates:
        raise SystemExit(f"no track: {args.cam} 连续 >={args.min_frames} 帧且全程 <={args.max_dist} m")
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    n, _, tk, have, dists = candidates[0]
    print(f"selected track={tk} cam={args.cam} frames={n} "
          f"f{have[0]['fidx']:03d}-f{have[-1]['fidx']:03d} dist={np.round(dists,1)} m")

    # ---- 逐帧：读注入后合成图画 bbox ----
    frames_rgb = []
    for f, dist in zip(have, dists):
        g = frame_gts(f)[f["oi"]]
        sd = meta.sd_for_sample(f["tok"], args.cam)
        cam_spec = {"K": np.asarray(meta.calib[sd["calibrated_sensor_token"]]["camera_intrinsic"], float),
                    "T_cam2global": np.linalg.inv(camera_channel.global2cam(sd, meta)),
                    "width": int(sd["width"]), "height": int(sd["height"])}
        img = cv2.imread(str(f["dir"] / "composite" / f"{args.cam}.jpg"))
        if img is None:
            print(f"  skip f{f['fidx']:03d}: no composite for {args.cam}")
            continue
        xyxy, _ = project_box_AABB(g, cam_spec)
        if xyxy is None:
            continue
        x0, y0, x1, y1 = [int(v) for v in xyxy]
        # 黄框 + 黑底标签（类目中文 + 距离）
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 4)
        for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            cv2.rectangle(img, (cx - 7, cy - 7), (cx + 7, cy + 7), (0, 255, 255), -1)
        cls = cn_map.get(Path(g["model_path"]).name,
                         Path(g["model_path"]).name.rsplit("_", 2)[0])
        label = f"OOD {cls}  {dist:.1f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        ty = max(y0 - 14, th + 10)
        cv2.rectangle(img, (x0, ty - th - 10), (x0 + tw + 12, ty + 8), (0, 0, 0), -1)
        cv2.putText(img, label, (x0 + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2, cv2.LINE_AA)
        frames_rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not frames_rgb:
        raise SystemExit("no drawable frame")

    # ---- 缩放 + 调色板 + GIF ----
    w = args.width
    h = int(frames_rgb[0].shape[0] * w / frames_rgb[0].shape[1])
    pil = [Image.fromarray(a).resize((w, h), Image.LANCZOS).quantize(
        colors=256, method=Image.MEDIANCUT) for a in frames_rgb]
    out = Path(args.out) if args.out else (
        # run_id 入名：不同类目运行可能选中同名 track（同 seed 同 scene），
        # 缺了它后一类目的 GIF 会覆盖前一类目的。
        ROOT / "data" / "outputs" / "OoD" / "ood_gif"
        / f"{args.run}__{tk}_{args.cam}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = int(1000 / args.fps)
    pil[0].save(out, save_all=True, append_images=pil[1:], duration=dur,
                loop=0, optimize=True)
    print(f"wrote {out}  ({len(pil)} frames, {dur} ms/frame, {out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
