#!/usr/bin/env python3
"""把 road_ood 模型池的 108 个 LVIS 类目划分为 ID / OOD。

判定口径（三档）：
  ID-CORE  映射到 nuScenes detection 评测的 10 类（devkit DETECTION_NAMES），
           即"常用目标"的客观操作定义 —— 训练/评测都覆盖，标注量大。
  ID-TAIL  在 nuScenes 完整税则（category.json 共 23 类）内，但极稀有：
           v1.0-mini 零标注或仅数十帧，不参与 detection 评测。
           注入后检测器"理论上该认"，但 practically 不会 —— 属灰区。
  OOD      不在 nuScenes 税则内 —— 真·类别 OOD。

输出：
  assets/models/road_ood_docum/road_ood_class_map.csv      逐类判定
  assets/models/road_ood_docum/road_ood_OOD_CLASSES.md     人类可读清单
  assets/models/road_ood_docum/road_ood_asset_tags.csv     逐文件 ID/OOD 标签（单一事实来源）

用法：
  python scripts/classify_id_ood.py                # 生成两份清单
  python scripts/classify_id_ood.py --report       # 只在终端打统计
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 资产统一平铺在 assets/models/，不再按 ID/OOD 分子目录；
# 语义判定完全由 road_ood_docum/ 下的文档承载。
ASSET_DIR = ROOT / "assets" / "models"
DOC_DIR = ASSET_DIR / "road_ood_docum"
PREFIX = "road_ood"
MANIFEST_CSV = DOC_DIR / f"{PREFIX}_manifest.csv"
CLASS_MAP_CSV = DOC_DIR / f"{PREFIX}_class_map.csv"
OOD_MD = DOC_DIR / f"{PREFIX}_OOD_CLASSES.md"
ASSET_TAGS_CSV = DOC_DIR / f"{PREFIX}_asset_tags.csv"

# ---------------------------------------------------------------- 判定表
# lvis_class -> (tier, nuscenes_category, reason)
# tier: ID-CORE / ID-TAIL / OOD
CORE = "ID-CORE"
TAIL = "ID-TAIL"
OOD = "OOD"

TC = "movable_object.trafficcone"
BAR = "movable_object.barrier"
PUSH = "movable_object.pushable_pullable"
DEB = "movable_object.debris"
RACK = "static_object.bicycle_rack"
BIC = "vehicle.bicycle"
MOT = "vehicle.motorcycle"
CON = "vehicle.construction"
ANI = "animal"
STR = "human.pedestrian.stroller"
WCH = "human.pedestrian.wheelchair"
PMB = "human.pedestrian.personal_mobility"

MAPPING: dict[str, tuple[str, str, str]] = {
    # ---------------- 道路设施 ----------------
    "cone": (CORE, TC, "nuScenes 有 movable_object.trafficcone，且属 detection 10 类"),
    "jersey": (CORE, BAR, "泽西护栏 = nuScenes movable_object.barrier 的典型形态"),
    "sawhorse": (CORE, BAR, "施工用锯木架即临时路障支架，归 barrier"),
    "dumpster": (TAIL, PUSH, "大型垃圾箱：项目已实测用 pushable_pullable 落库；该类在 mini 仅 82 帧"),
    "trash_can": (TAIL, PUSH, "垃圾桶归 pushable_pullable；mini 仅 82 帧（全类合计）"),
    "awning": (OOD, "", "路边固定设施，nuScenes 不标注静态路面附属物"),
    "bench": (OOD, "", "路边固定设施，非交通参与者"),
    "billboard": (OOD, "", "固定广告牌，nuScenes 不标注"),
    "chair": (OOD, "", "家具类固定物"),
    "fire_alarm": (OOD, "", "建筑附属，尺寸过小"),
    "fire_extinguisher": (OOD, "", "建筑附属，尺寸过小"),
    "fireplug": (OOD, "", "消防栓：KITTI-OoD 常见，但 nuScenes 无此税则"),
    "flagpole": (OOD, "", "杆状固定设施"),
    "garbage": (OOD, "", "散落垃圾，尺寸远小于 debris 的下限"),
    "lamppost": (OOD, "", "杆状固定设施（nuScenes 只标可移动物/车/人）"),
    "mail_slot": (OOD, "", "建筑附属，尺寸过小"),
    "mailbox_(at_home)": (OOD, "", "私人信箱，尺寸过小"),
    "manhole": (OOD, "", "嵌于路面，无法作为独立立体目标注入"),
    "milestone": (OOD, "", "路侧小构件"),
    "newsstand": (OOD, "", "固定售卖亭"),
    "parking_meter": (OOD, "", "路侧小构件"),
    "pole": (OOD, "", "杆状固定设施"),
    "postbox_(public)": (OOD, "", "公共邮筒，nuScenes 不标注"),
    "signboard": (OOD, "", "店招/标牌，非交通标志"),
    "silo": (OOD, "", "工业建筑体"),
    "stop_sign": (OOD, "", "停车标志：KITTI-OoD 高频，nuScenes 明确不标交通标志"),
    "street_sign": (OOD, "", "路牌，nuScenes 不标注"),
    "streetlight": (OOD, "", "杆状固定设施"),
    "telephone_booth": (OOD, "", "固定亭体"),
    "telephone_pole": (OOD, "", "杆状固定设施"),
    "traffic_light": (OOD, "", "信号灯：nuScenes 不标注（依赖地图/先验）"),
    "vending_machine": (OOD, "", "固定设备"),
    "water_tower": (OOD, "", "工业建筑体"),
    # ---------------- 遗撒物 / 杂物 ----------------
    "barrel": (TAIL, DEB, "油桶：可移动、留在可行驶面上 —— debris 的定义"),
    "box": (TAIL, DEB, "纸箱遗撒，debris 定义内"),
    "bucket": (TAIL, DEB, "桶类遗撒，debris 定义内"),
    "crate": (TAIL, DEB, "板条箱遗撒，debris 定义内"),
    "ladder": (TAIL, DEB, "梯子掉落，debris 定义内"),
    "log": (TAIL, DEB, "原木掉落，debris 定义内"),
    "mattress": (TAIL, DEB, "床垫遗撒，debris 定义内"),
    "pipe": (TAIL, DEB, "管材遗撒，debris 定义内"),
    "tarp": (TAIL, DEB, "篷布遗撒（可覆盖路面），debris 定义内"),
    "barrow": (TAIL, PUSH, "独轮手推车 = 推/拉对象"),
    "handcart": (TAIL, PUSH, "手推车 = 推/拉对象"),
    "cooler_(for_food)": (TAIL, PUSH, "保温箱：可推拉的小件"),
    "suitcase": (TAIL, PUSH, "行李箱是 nuScenes pushable_pullable 的教科书例子"),
    "armoire": (OOD, "", "整体衣柜，体量与路面遗撒物不符"),
    "automatic_washer": (OOD, "", "家电，nuScenes 无对应税则"),
    "bathtub": (OOD, "", "卫浴大件，nuScenes 无对应税则"),
    "bed": (OOD, "", "大件家具"),
    "bookcase": (OOD, "", "大件家具"),
    "cabinet": (OOD, "", "大件家具"),
    "plastic_bag": (OOD, "", "尺寸过小，不足以构成可检测立体目标"),
    "refrigerator": (OOD, "", "家电，nuScenes 无对应税则"),
    "sink": (OOD, "", "卫浴件"),
    "sofa": (OOD, "", "大件家具"),
    "table": (OOD, "", "大件家具"),
    "toilet": (OOD, "", "卫浴件"),
    # ---------------- 动物 ----------------
    "bear": (TAIL, ANI, "nuScenes 确有 animal 类目，但全库仅个位数实例"),
    "camel": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "cat": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "chicken_(animal)": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "cow": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "deer": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "dog": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "domestic_ass": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "duck": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "elephant": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "frog": (TAIL, ANI, "同上：尺寸过小，animal 税则内"),
    "giraffe": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "goat": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "goose": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "hippopotamus": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "hog": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "horse": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "lion": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "monkey": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "rabbit": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "rhinoceros": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "sheep": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "snake": (TAIL, ANI, "同上：尺寸过小，animal 税则内"),
    "squirrel": (TAIL, ANI, "同上：尺寸过小，animal 税则内"),
    "tiger": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    "turtle": (TAIL, ANI, "同上：尺寸过小，animal 税则内"),
    "zebra": (TAIL, ANI, "同上：animal 税则内，极稀有"),
    # ---------------- 非典型交通参与者 ----------------
    "bicycle": (CORE, BIC, "nuScenes 有 vehicle.bicycle，属 detection 10 类"),
    "motorcycle": (CORE, MOT, "nuScenes 有 vehicle.motorcycle，属 detection 10 类"),
    "motor_scooter": (TAIL, MOT, "电动踏板/小排量踏板车，归 motorcycle 但形态不标准"),
    "baby_buggy": (TAIL, STR, "nuScenes 有 human.pedestrian.stroller，mini 零标注"),
    "skateboard": (TAIL, PMB, "nuScenes 有 human.pedestrian.personal_mobility，mini 25 帧"),
    "wheelchair": (TAIL, WCH, "nuScenes 有 human.pedestrian.wheelchair，mini 零标注"),
    "shopping_cart": (TAIL, PUSH, "推车 = pushable_pullable 的典型例子"),
    "boat": (OOD, "", "船舶，nuScenes 城市场景无此税则"),
    "canoe": (OOD, "", "独木舟，非道路目标"),
    "golfcart": (OOD, "", "场地车，nuScenes 无对应税则"),
    "horse_buggy": (OOD, "", "马车，nuScenes 无对应税则"),
    "horse_carriage": (OOD, "", "马车，nuScenes 无对应税则"),
    "kayak": (OOD, "", "皮划艇，非道路目标"),
    "sled": (OOD, "", "雪橇，非道路目标"),
    "snowmobile": (OOD, "", "雪地摩托，nuScenes 无对应税则"),
    "surfboard": (OOD, "", "冲浪板，非道路目标"),
    # ---------------- 施工 / 工业 ----------------
    "bulldozer": (CORE, CON, "nuScenes vehicle.construction 的典型实例"),
    "forklift": (CORE, CON, "nuScenes vehicle.construction 定义覆盖"),
    "power_shovel": (CORE, CON, "挖掘机 = vehicle.construction 的典型实例"),
    "generator": (OOD, "", "发电机组：非车辆、非路面可移动物"),
    "lawn_mower": (OOD, "", "割草机：可移动但 nuScenes 无此税则"),
    "plow_(farm_equipment)": (OOD, "", "农具：nuScenes construction 限定为施工用途"),
    "tractor_(farm_equipment)": (OOD, "", "农用拖拉机：nuScenes construction 限定为施工用途"),
}

# 机翻错名修正（manifest 里的 class_cn 有几处明显错误）
CN_FIX = {
    "jersey": "泽西护栏/水泥隔离墩",
    "domestic_ass": "驴",
    "cone": "交通锥",
    "sawhorse": "锯木架/施工支架",
    "power_shovel": "挖掘机",
    "barrow": "独轮手推车",
    "motor_scooter": "电动踏板车",
    "garbage": "垃圾/垃圾袋",
    "pipe": "管材",
}

TIER_ORDER = {CORE: 0, TAIL: 1, OOD: 2}
TIER_CN = {CORE: "ID 主集（detection 10 类）", TAIL: "ID 长尾（23 类税则内）", OOD: "OOD（税则外）"}


def load_pool() -> tuple[dict, dict]:
    """返回 (每类统计, 每类中文名)"""
    rows = list(csv.DictReader(open(MANIFEST_CSV, encoding="utf-8")))
    stat: dict[str, dict] = {}
    for r in rows:
        s = stat.setdefault(
            r["lvis_class"],
            {"group": r["group"], "group_cn": r["group_cn"], "n": 0,
             "mb": 0.0, "materials": 0, "textures": 0, "cn": r["class_cn"]},
        )
        s["n"] += 1
        s["mb"] += float(r["size_MB"])
        s["materials"] += int(r["materials"])
        s["textures"] += int(r["textures"])
    return stat, {k: v["cn"] for k, v in stat.items()}



def write_asset_tags(out_rows) -> tuple[list, Counter]:
    """给 assets/models/ 下每个 .glb 打上 ID/OOD 标签。

    平铺布局里目录不再表达语义，这份表就是「某个资产是否 OOD」的单一事实来源。
    判定靠 sidecar 的 `lvis_class` 关联类目表；未登记类目一律按 OOD 处理。
    """
    cls2row = {r["lvis_class"]: r for r in out_rows}
    tags, unmapped = [], Counter()
    for glb in sorted(ASSET_DIR.glob("*.glb")):
        side = glb.with_suffix(".json")
        d = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
        cls = d.get("lvis_class")
        r = cls2row.get(cls)
        if r:
            tier, ncat = r["tier"], r["nuscenes_category"]
            cn, group = r["class_cn"], r["group"]
        else:
            tier, ncat = OOD, ""
            cn = d.get("class_cn") or "(未登记类目)"
            group = "legacy"
            unmapped[str(cls)] += 1
        is_ood = 0 if tier == CORE else 1
        tags.append({
            "file": glb.name,
            "lvis_class": cls or "",
            "class_cn": cn,
            "group": group,
            "tier": tier,
            "is_ood": is_ood,
            "in_nuscenes_taxonomy": 0 if tier == OOD else 1,
            "nuscenes_category": ncat,
            "ood_category": "movable_object.anomaly" if is_ood else "",
            "license": d.get("license") or "unknown",
            "author": d.get("author") or "",
            "uid": d.get("uid") or "",
            "size_MB": f"{glb.stat().st_size / 1e6:.2f}",
        })
    ASSET_TAGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(ASSET_TAGS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tags[0].keys()))
        w.writeheader()
        w.writerows(tags)
    return tags, unmapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="只在终端输出统计")
    args = ap.parse_args()

    stat, cn = load_pool()

    missing = sorted(set(stat) - set(MAPPING))
    extra = sorted(set(MAPPING) - set(stat))
    if missing or extra:
        raise SystemExit(f"判定表与模型池不匹配\n  未判定: {missing}\n  多余: {extra}")

    out_rows = []
    for cls, s in sorted(stat.items(), key=lambda kv: (TIER_ORDER[MAPPING[kv[0]][0]], kv[1]["group"], kv[0])):
        tier, ncat, reason = MAPPING[cls]
        out_rows.append({
            "lvis_class": cls,
            "class_cn": CN_FIX.get(cls, s["cn"]),
            "group": s["group"],
            "group_cn": s["group_cn"],
            "tier": tier,
            "nuscenes_category": ncat,
            "n_models": s["n"],
            "size_MB": round(s["mb"], 2),
            "materials": s["materials"],
            "textures": s["textures"],
            "reason": reason,
        })

    # ---- 统计
    tier_cnt = Counter(r["tier"] for r in out_rows)
    tier_models = Counter()
    tier_mb = defaultdict(float)
    for r in out_rows:
        tier_models[r["tier"]] += r["n_models"]
        tier_mb[r["tier"]] += r["size_MB"]

    print(f"{'档位':<8}{'类目':>5}{'模型':>7}{'占比':>8}{'体积GB':>9}")
    total_m = sum(tier_models.values())
    for t in (CORE, TAIL, OOD):
        print(f"{t:<8}{tier_cnt[t]:>5}{tier_models[t]:>7}"
              f"{tier_models[t]/total_m*100:>7.1f}%{tier_mb[t]/1024:>9.2f}")
    print(f"{'合计':<8}{len(out_rows):>5}{total_m:>7}{100:>7.1f}%{sum(tier_mb.values())/1024:>9.2f}")

    if args.report:
        for t in (CORE, TAIL, OOD):
            print(f"\n== {t} ({TIER_CN[t]}) ==")
            for r in out_rows:
                if r["tier"] == t:
                    print(f"  {r['lvis_class']:<24}{r['class_cn']:<12}{r['n_models']:>4}  {r['nuscenes_category'] or '-'}")
        return

    # ---- 写 CSV
    csv_path = CLASS_MAP_CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # ---- 写 Markdown
    md = [
        "# ID / OOD 类目划分（108 类）",
        "",
        "判定口径见 `scripts/classify_id_ood.py` 头部注释。依据：本地 nuScenes v1.0-mini 的",
        "`category.json`（23 类全税则）+ devkit `DETECTION_NAMES`（10 类评测口径）。",
        "",
        "| 档位 | 类目 | 模型 | 占比 | 体积 |",
        "|---|---|---|---|---|",
    ]
    for t in (CORE, TAIL, OOD):
        md.append(f"| `{t}` {TIER_CN[t]} | {tier_cnt[t]} | {tier_models[t]} | "
                  f"{tier_models[t]/total_m*100:.1f}% | {tier_mb[t]/1024:.2f} GB |")
    md.append(f"| **合计** | **{len(out_rows)}** | **{total_m}** | 100% | "
              f"**{sum(tier_mb.values())/1024:.2f} GB** |")
    md.append("")

    for t in (CORE, TAIL, OOD):
        sub = [r for r in out_rows if r["tier"] == t]
        md.append(f"## {t} — {TIER_CN[t]}（{len(sub)} 类 / {tier_models[t]} 个）")
        md.append("")
        md.append("| LVIS 类目 | 中文 | 分组 | 模型数 | 建议注入 category | 依据 |")
        md.append("|---|---|---|---|---|---|")
        for r in sub:
            md.append(f"| `{r['lvis_class']}` | {r['class_cn']} | {r['group_cn']} | {r['n_models']} | "
                      f"`{r['nuscenes_category'] or 'movable_object.anomaly'}` | {r['reason']} |")
        md.append("")

    OOD_MD.write_text("\n".join(md), encoding="utf-8")

    # ---- 逐文件标签（平铺布局下判定 OOD 的唯一依据）
    tags, unmapped = write_asset_tags(out_rows)
    n_ood = sum(t["is_ood"] for t in tags)
    print(f"\n逐文件标签: {len(tags)} 个资产，OOD {n_ood} 个 / ID {len(tags)-n_ood} 个")
    if unmapped:
        print("未登记类目（按 OOD 兜底）:", dict(unmapped))
    print(f"wrote {csv_path}")
    print(f"wrote {OOD_MD}")
    print(f"wrote {ASSET_TAGS_CSV}")


if __name__ == "__main__":
    main()
