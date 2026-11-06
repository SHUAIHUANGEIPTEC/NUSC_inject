#!/usr/bin/env python3
"""Download road-scene OOD targets from Objaverse v1 that carry real PBR textures.

Pipeline
  1. LVIS class whitelist (5 groups, see GROUPS) -> candidate UIDs
  2. metadata filter: license in {by, cc0}, isDownloadable, archives.glb.textureCount > 0
  3. size cap + round-robin across classes so every class gets covered
  4. download .glb, re-parse its JSON chunk, keep only files that truly contain
     materials + textures (metadata textureCount is a Sketchfab hint, not a guarantee)
  5. .glb + sidecar land FLAT in assets/models/ (assets are unified);
     MANIFEST.md + manifest.csv + skipped.csv + candidates.json live in
     assets/models/road_ood_docum/ with a `road_ood_` prefix

All assets come from hf-mirror.com: no HF token, no `objaverse` package.

Usage:
  python scripts/download_road_ood.py --report-only
  python scripts/download_road_ood.py --max-total-gb 25 --workers 8
  python scripts/download_road_ood.py --groups animals,road_facilities --max-per-class 10
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://hf-mirror.com/datasets/allenai/objaverse/resolve/main"
CACHE = Path.home() / ".cache" / "objaverse-mirror"
ROOT = Path(__file__).resolve().parents[1]
# 资产统一平铺：不按 ID/OOD 分子目录，语义完全由 road_ood_docum/ 的文档约束
ASSET_DIR = ROOT / "assets" / "models"
DOC_DIR = ASSET_DIR / "road_ood_docum"
PREFIX = "road_ood"
ASSET_LABEL = "assets/models/"
DEFAULT_OUT = ASSET_DIR


def doc_paths(out_dir: Path | None = None) -> dict[str, Path]:
    """清单固定写到文档目录，不受 `--out` 影响（资产目录里只留 .glb + sidecar）。"""
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "manifest_md": DOC_DIR / f"{PREFIX}_MANIFEST.md",
        "manifest_csv": DOC_DIR / f"{PREFIX}_manifest.csv",
        "skipped_csv": DOC_DIR / f"{PREFIX}_skipped.csv",
        "candidates": DOC_DIR / f"{PREFIX}_candidates.json",
    }

GROUPS = {
    "road_facilities": ("道路设施", [
        "traffic_light", "street_sign", "stop_sign", "signboard", "cone", "jersey",
        "fireplug", "parking_meter", "mailbox_(at_home)", "postbox_(public)",
        "mail_slot", "bench", "chair", "trash_can", "dumpster", "garbage",
        "telephone_pole", "pole", "streetlight", "lamppost", "flagpole", "manhole",
        "milestone", "awning", "billboard", "newsstand", "telephone_booth",
        "vending_machine", "fire_alarm", "fire_extinguisher", "water_tower", "silo",
    ]),
    "dropped_items": ("遗撒物/杂物", [
        "crate", "barrel", "box", "suitcase", "ladder", "mattress", "refrigerator",
        "automatic_washer", "sofa", "table", "bed", "armoire", "cabinet", "bookcase",
        "bathtub", "toilet", "sink", "bucket", "cooler_(for_food)", "sawhorse",
        "handcart", "barrow", "plastic_bag", "tarp", "pipe", "log",
    ]),
    "animals": ("动物", [
        "dog", "cat", "deer", "cow", "horse", "sheep", "goat", "hog", "elephant",
        "bear", "lion", "tiger", "camel", "domestic_ass", "chicken_(animal)", "duck",
        "goose", "rabbit", "squirrel", "frog", "snake", "turtle", "zebra", "giraffe",
        "hippopotamus", "rhinoceros", "monkey",
    ]),
    "atypical_actors": ("非典型交通参与者", [
        "shopping_cart", "baby_buggy", "wheelchair", "skateboard", "bicycle",
        "motorcycle", "motor_scooter", "kayak", "canoe", "boat", "surfboard",
        "horse_carriage", "horse_buggy", "sled", "snowmobile", "golfcart",
    ]),
    "construction": ("施工/工业设备", [
        "bulldozer", "forklift", "tractor_(farm_equipment)", "plow_(farm_equipment)",
        "power_shovel", "lawn_mower", "generator",
    ]),
}

LICENSE_URL = {
    "by": "https://creativecommons.org/licenses/by/4.0/",
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
}


def fetch(rel: str) -> Path:
    dst = CACHE / rel
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    cmd = ["curl", "-sSL", "--fail", "--retry", "3", "--max-time", "300",
           "-o", str(part), f"{BASE}/{rel}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not part.exists() or part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {rel} :: {r.stderr.strip()[:200]}")
    part.replace(dst)
    return dst


def load_gz_json(rel: str):
    return json.load(gzip.open(fetch(rel), "rt", encoding="utf-8"))


def glb_stats(path: Path):
    """Parse the glTF JSON chunk; None if not a valid glb."""
    try:
        with open(path, "rb") as f:
            head = f.read(20)
            if len(head) < 20 or head[:4] != b"glTF" or head[16:20] != b"JSON":
                return None
            raw = f.read(int.from_bytes(head[12:16], "little"))
        g = json.loads(raw.decode("utf-8").rstrip(" \t\r\n\0"))
    except Exception:
        return None
    mats = g.get("materials") or []
    return {"materials": len(mats),
            "images": len(g.get("images") or []),
            "textures": len(g.get("textures") or [])}


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip()).strip("_") or "model"


def build_candidates(groups, allowed, max_mb, min_mb):
    lvis = load_gz_json("lvis-annotations.json.gz")
    paths = load_gz_json("object-paths.json.gz")
    cn_path = CACHE / "lvis_classes_cn.json"
    cn = json.loads(cn_path.read_text()) if cn_path.exists() else {}

    wanted = {}
    for g in groups:
        for c in GROUPS[g][1]:
            for uid in lvis.get(c, []):
                wanted.setdefault(uid, (g, c))
    ncls = sum(len(GROUPS[g][1]) for g in groups)
    print(f"候选 uid: {len(wanted)}（{ncls} 个类目 / {len(groups)} 组）")

    shard_of = {u: paths[u].split("/")[1] for u in wanted}
    shards = sorted(set(shard_of.values()))
    print(f"扫描 {len(shards)} 个元数据分片 ...")

    rows = []
    for i, shard in enumerate(shards, 1):
        try:
            meta = load_gz_json(f"metadata/{shard}.json.gz")
        except Exception as e:
            print(f"  ! {shard}: {e}")
            continue
        for uid, (g, c) in wanted.items():
            if shard_of[uid] != shard:
                continue
            m = meta.get(uid)
            if not m or m.get("license") not in allowed or m.get("isDownloadable") is False:
                continue
            a = (m.get("archives") or {}).get("glb") or {}
            if (a.get("textureCount") or 0) <= 0:
                continue
            est_mb = (a.get("size") or 0) / 3.0 / 1e6
            if (max_mb and est_mb > max_mb) or (min_mb and est_mb < min_mb):
                continue
            user = m.get("user") or {}
            rows.append({
                "uid": uid, "group": g, "lvis_class": c,
                "class_cn": cn.get(c, c), "name": m.get("name") or uid,
                "license": m.get("license"),
                "author": user.get("displayName") or user.get("username"),
                "path": paths[uid], "vertexCount": m.get("vertexCount"),
                "faceCount": m.get("faceCount"), "textureCount": a.get("textureCount"),
                "textureMaxResolution": a.get("textureMaxResolution"),
                "estRealMB": est_mb,
            })
        if i % 20 == 0:
            print(f"  {i}/{len(shards)} 片，命中 {len(rows)}")
    print(f"元数据筛选后: {len(rows)} 个候选（CC-BY/CC0 + 有贴图）")
    return rows


def allocate(rows, max_per_class, max_total_gb):
    """Round-robin across classes: every class gets one before any gets two."""
    by_cls = {}
    for r in rows:
        by_cls.setdefault(r["lvis_class"], []).append(r)
    for k in by_cls:
        by_cls[k].sort(key=lambda r: r["estRealMB"])
    order = sorted(by_cls)
    idx = {k: 0 for k in order}
    picked, used = [], 0.0
    budget_mb = max_total_gb * 1000
    while True:
        moved = False
        for k in order:
            if (max_per_class and idx[k] >= max_per_class) or idx[k] >= len(by_cls[k]):
                continue
            r = by_cls[k][idx[k]]
            if picked and used + r["estRealMB"] > budget_mb:
                continue
            picked.append(r)
            used += r["estRealMB"]
            idx[k] += 1
            moved = True
        if not moved:
            break
    return picked


def download_one(r, out_dir: Path, seq: int):
    stem = f"{slug(r['lvis_class'])}_{seq:03d}_{r['uid'][:8]}"
    glb = out_dir / f"{stem}.glb"
    if glb.exists() and glb.stat().st_size > 0:
        st = glb_stats(glb) or {}
        return {**r, "file": glb.name, "status": "exists",
                "size": glb.stat().st_size,
                "materials": st.get("materials", 0), "textures": st.get("textures", 0)}
    tmp = out_dir / f".{stem}.part"
    p = subprocess.run(["curl", "-sSL", "--fail", "--retry", "2", "--max-time", "180",
                        "-o", str(tmp), f"{BASE}/{r['path']}"],
                       capture_output=True, text=True)
    if p.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return {**r, "file": None, "status": "download_failed"}
    st = glb_stats(tmp)
    if not st:
        tmp.unlink(missing_ok=True)
        return {**r, "file": None, "status": "invalid_glb"}
    if st["materials"] == 0 or st["textures"] == 0:
        tmp.unlink(missing_ok=True)
        return {**r, "file": None, "status": "no_pbr_texture"}
    tmp.replace(glb)
    (out_dir / f"{stem}.json").write_text(json.dumps({
        "uid": r["uid"], "lvis_class": r["lvis_class"], "class_cn": r["class_cn"],
        "group": r["group"], "name": r["name"], "license": r["license"],
        "license_url": LICENSE_URL.get(r["license"]),
        "sketchfab_uri": f"https://api.sketchfab.com/v3/models/{r['uid']}",
        "author": r["author"], "path": r["path"], "vertexCount": r["vertexCount"],
        "faceCount": r["faceCount"], "materialCount": st["materials"],
        "textureCount": st["textures"],
        "textureMaxResolution": r["textureMaxResolution"],
        "source": "Objaverse v1 / allenai/objaverse (via hf-mirror.com)",
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**r, "file": glb.name, "status": "ok", "size": glb.stat().st_size,
            "materials": st["materials"], "textures": st["textures"]}


def write_manifest(ok_rows, skip_rows, out_dir: Path, groups):
    D = doc_paths(out_dir)
    with open(D["manifest_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "group", "group_cn", "lvis_class", "class_cn", "name",
                    "license", "author", "uid", "vertexCount", "faceCount",
                    "materials", "textures", "size_MB"])
        for r in sorted(ok_rows, key=lambda x: (x["group"], x["lvis_class"], x["file"])):
            w.writerow([r["file"], r["group"], GROUPS[r["group"]][0], r["lvis_class"],
                        r["class_cn"], r["name"].replace("\n", " "), r["license"],
                        r["author"], r["uid"], r["vertexCount"], r["faceCount"],
                        r["materials"], r["textures"], f"{r['size']/1e6:.2f}"])
    if skip_rows:
        with open(D["skipped_csv"], "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["uid", "group", "lvis_class", "name", "status", "est_MB", "path"])
            for r in skip_rows:
                w.writerow([r["uid"], r["group"], r["lvis_class"],
                            r["name"].replace("\n", " "), r["status"],
                            f"{r['estRealMB']:.2f}", r["path"]])

    import datetime
    total_mb = sum(r["size"] for r in ok_rows) / 1e6
    L = ["# 道路交通场景 OOD 目标 —— 下载清单", "",
         f"- 生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M}",
         f"- 资产目录：`{ASSET_LABEL}`（统一平铺，仅 `.glb` + 同名 sidecar）",
         f"- 成功下载：**{len(ok_rows)}** 个 glb，合计 **{total_mb/1024:.2f} GB**",
         f"- 跳过/失败：{len(skip_rows)} 个（明细见 `{D['skipped_csv'].name}`）",
         "- 许可：仅 `by`(CC-BY 4.0) 与 `cc0`；逐文件署名见同名 `.json` sidecar",
         "- 所有文件均已二次校验：glTF 合法 + 至少 1 个 material + 至少 1 个 texture",
         "", "## 按分组统计", "", "| 分组 | 类目数 | 模型数 | 体积 |", "|---|---|---|---|"]
    for g in groups:
        s = [r for r in ok_rows if r["group"] == g]
        if s:
            L.append(f"| {GROUPS[g][0]} `{g}` | {len({r['lvis_class'] for r in s})} | {len(s)} | "
                     f"{sum(r['size'] for r in s)/1e9:.2f} GB |")
    L += ["", "## 按类目明细", "", "| LVIS 类目 | 中文 | 分组 | 数量 | 体积 |", "|---|---|---|---|---|"]
    for g in groups:
        by = {}
        for r in ok_rows:
            if r["group"] == g:
                by.setdefault(r["lvis_class"], []).append(r)
        for c in sorted(by):
            rs = by[c]
            L.append(f"| `{c}` | {rs[0]['class_cn']} | {GROUPS[g][0]} | {len(rs)} | "
                     f"{sum(x['size'] for x in rs)/1e6:.1f} MB |")
    L += ["", f"## 文件列表（相对 `{ASSET_LABEL}`）", "",
          "| 文件 | LVIS 类目 | 材质 | 贴图 | MB | 作者 | 许可 |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(ok_rows, key=lambda x: (x["group"], x["lvis_class"], x["file"])):
        L.append(f"| `{r['file']}` | `{r['lvis_class']}` | {r['materials']} | {r['textures']} | "
                 f"{r['size']/1e6:.1f} | {r['author']} | {r['license']} |")
    D["manifest_md"].write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"清单: {D['manifest_md']} / {D['manifest_csv'].name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--groups", default="all", help="all 或逗号分隔：" + ",".join(GROUPS))
    ap.add_argument("--license", default="by,cc0")
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--max-mb", type=float, default=25.0)
    ap.add_argument("--min-mb", type=float, default=0.05)
    ap.add_argument("--max-total-gb", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    groups = list(GROUPS) if args.groups == "all" else [g.strip() for g in args.groups.split(",")]
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        sys.exit(f"unknown group(s): {bad}")

    rows = build_candidates(groups, {x.strip() for x in args.license.split(",") if x.strip()},
                            args.max_mb, args.min_mb)
    if not rows:
        sys.exit("没有符合条件的目标")
    picked = allocate(rows, args.max_per_class, args.max_total_gb)
    print(f"\n入选 {len(picked)} 个 / {len({r['lvis_class'] for r in picked})} 类，"
          f"预估 {sum(r['estRealMB'] for r in picked)/1000:.1f} GB")
    for g in groups:
        s = [r for r in picked if r["group"] == g]
        if s:
            print(f"  {GROUPS[g][0]:12s} {len(s):5d} 个  "
                  f"{sum(x['estRealMB'] for x in s)/1000:6.2f} GB")

    out_dir = Path(args.out)
    if args.report_only:
        D = doc_paths(out_dir)
        D["candidates"].write_text(json.dumps(picked, indent=1, ensure_ascii=False))
        print(f"\n候选写入 {D['candidates']}（未下载）")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n下载到 {out_dir}（{args.workers} 并发）...")
    seq = {r["uid"]: i for i, r in
           enumerate(sorted(picked, key=lambda x: (x["group"], x["lvis_class"], x["uid"])), 1)}
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, r, out_dir, seq[r["uid"]]) for r in picked]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 50 == 0 or i == len(picked):
                ok = sum(1 for x in results if x["status"] in ("ok", "exists"))
                print(f"  {i}/{len(picked)}  成功 {ok}")

    ok_rows = [r for r in results if r["status"] in ("ok", "exists")]
    skip_rows = [r for r in results if r["status"] not in ("ok", "exists")]
    print("\n结果:", dict(Counter(r["status"] for r in results)))
    write_manifest(ok_rows, skip_rows, out_dir, groups)
    print(f"完成：{len(ok_rows)} 个，{sum(r['size'] for r in ok_rows)/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
