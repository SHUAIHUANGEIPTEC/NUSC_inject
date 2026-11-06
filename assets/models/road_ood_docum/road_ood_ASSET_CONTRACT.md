# 资产契约：什么算资产、什么算 OOD

> 这份文档是 `assets/models/` 的**唯一规范**。资产统一平铺，目录结构不表达任何语义；
> 一个 `.glb` 是不是资产、是不是 OOD，一律以下面的定义和清单为准。
> 规范可执行部分在 `scripts/classify_id_ood.py`（判定表）与
> `scripts/download_road_ood.py`（下载/校验），本文档不重复实现，只固定口径。

## 1. 目录约定

```
assets/models/
├── <name>.glb                  资产本体，全部平铺在这一层，不建 ID/OOD 子目录
├── <name>.json                 同名 sidecar（元数据 + CC-BY 署名依据）
├── README.md                   原有 12 个模型的说明
├── .gitignore                  按命名模式忽略资产（见 §6）
└── road_ood_docum/             本目录：所有清单、规范、中间产物
    ├── road_ood_ASSET_CONTRACT.md   本文档
    ├── road_ood_class_map.csv       类目级 ID/OOD 判定（108 类）
    ├── road_ood_asset_tags.csv      逐文件 ID/OOD 标签（3,498 行，单一事实来源）
    ├── road_ood_MANIFEST.md         下载清单（分组统计 + 逐文件表）
    ├── road_ood_manifest.csv        同上，机器可读
    ├── road_ood_OOD_CLASSES.md      判定依据的可读版
    ├── road_ood_skipped.csv         跳过/失败明细
    └── road_ood_candidates.json     6,560 个候选 uid
```

**约定：资产层只放 `*.glb` + 同名 `*.json` + `README.md` + `.gitignore`；
文档一律进 `road_ood_docum/`，文件名加 `road_ood_` 前缀。**
ID 与 OOD **不分区存放** —— 混放是有意的，靠 §4 的标签区分。

## 2. 什么算一个合法资产

必须同时满足：

| 条件 | 要求 |
|---|---|
| 本体 | `*.glb`，合法 glTF 2.0 binary |
| 材质 | 解析 JSON chunk 后 `materials >= 1` **且** `textures >= 1`（自带内嵌 PBR 贴图；Sketchfab 元数据里的 `textureCount` 只是提示，不作数） |
| sidecar | 同名 `*.json`，至少含 `uid`、`lvis_class`、`license`、`author`、`path`、`source` |
| 许可 | `license ∈ {by, cc0}`；`by` 必须保留 sidecar 里的 `author` 用于 CC-BY 署名 |
| 来源 | Objaverse v1（`allenai/objaverse`，原始为 Sketchfab 资产）。**没有 CARLA 内置资产**，CARLA 只是上游项目的背景生成器 |
| 命名 | 新池：`<lvis_class>_<序号>_<uid前8位>.glb`；原有 12 个保留语义名（如 `traffic_cone_02.glb`） |
| 体积 | 单文件 ≤ 25 MB（下载时硬上限） |

当前规模：**3,498 个 glb / 18 GB**，其中 3,486 个来自本次批量下载、
11 个是项目原有 Objaverse 模型、`temple.glb` 是许可不明的外部遗留文件
（`license: unknown`，需要干净许可链时应替换掉）。

## 3. OOD 判定依据

判定锚点是本机可核对的两份权威数据，不靠记忆：

- nuScenes 的 `v1.0-mini/category.json`（即 `data/nuscenes/v1.0-mini/category.json`）—— **23 类**完整类别体系
- devkit `nuscenes/eval/detection/constants.py` 的 `DETECTION_NAMES` —— **10 类**评测类别
  （car / truck / bus / trailer / construction_vehicle / pedestrian / motorcycle /
   bicycle / traffic_cone / barrier），即「常用目标」的客观操作定义

三档：

| 档位 | 含义 | 类目 | 模型 |
|---|---|---|---|
| `ID-CORE` | 能映射到 10 类评测类别，训练/评测都覆盖 | 8 | 244 |
| `ID-TAIL` | 在 23 类税则内但极稀有（mini 中 `animal` 零标注、`debris` 13 帧、`pushable_pullable` 82 帧） | 47 | 1,570 |
| `OOD` | 不在 nuScenes 类别体系内 | 53 | 1,672 |

**本项目的二分口径（用户定义）**：ID = 常用的 `ID-CORE`；**ID 之外全部算 OOD**，
即 `ID-TAIL` 也计入 OOD。理由：检测器在 nuScenes 上训练后不会认识大象，
把 890 个动物算成 ID 会让 OOD 实验失去意义。

判定流程：读 sidecar 的 `lvis_class` → 查 `road_ood_class_map.csv` → 得到 `tier`
→ `is_ood = 0 if tier == "ID-CORE" else 1`。**未登记类目一律按 OOD 兜底**
（当前只有 `temple.glb` 一例）。

全套 3,498 个文件的结果：**OOD 3,252 个 / ID 246 个**。
（246 = 池内 244 个 + 两个原有的 `cone` / `jersey` 模型。）

## 4. 单一事实来源

| 问题 | 去哪查 |
|---|---|
| 某个**类目**是 ID 还是 OOD、依据是什么 | `road_ood_class_map.csv` 的 `tier` / `nuscenes_category` / `reason` |
| 某个**文件**是不是 OOD、落库该用什么 category | `road_ood_asset_tags.csv` 的 `is_ood` / `ood_category` |
| 可读版、逐类理由 | `road_ood_OOD_CLASSES.md` |
| 作者、许可、材质贴图数、体积 | `road_ood_manifest.csv` + 同名 sidecar |

`road_ood_asset_tags.csv` 关键列：

- `is_ood`：1 = OOD（本项目口径），0 = ID
- `in_nuscenes_taxonomy`：1 = 在 23 类税则内（含 `ID-TAIL`），可用于宽松口径对照
- `nuscenes_category`：语义上对应的 nuScenes 类目（OOD 为空）
- `ood_category`：OOD 目标**建议落库类别**，固定 `movable_object.anomaly`

查某个文件：

```bash
grep '^Cat.glb,' assets/models/road_ood_docum/road_ood_asset_tags.csv
```

重新生成（改动判定表后必跑）：

```bash
python scripts/classify_id_ood.py
```

## 5. 落库时的注意事项

- **类别 ID ≠ 实例正常**。246 个 ID 资产的类别正常，但模型形态/姿态可以很异常，
  这属于「上下文 OOD」，要和「类别 OOD」分开设计实验。
- **要测 OOD 拒识，就不要写真实 category**。若把注入目标落库为
  `movable_object.trafficcone`，下游会把它当 ID 输出，测到的是召回而不是拒识。
  OOD 目标统一用 `movable_object.anomaly`（`configs/p0.yaml` 默认值），
  真实语义另存属性字段。
- 判定表给的是**语义归属**，不是强制的落库值；做「ID 召回」对照实验时，
  才用 `nuscenes_category` 里的真实类目。

## 6. 新增资产 / git 边界

新增一个资产的步骤：下载 glb → 写同名 sidecar（字段见 §2）→ 在
`classify_id_ood.py` 的 `MAPPING` 里登记 `lvis_class`（未登记会被兜底成 OOD）
→ 重跑 `python scripts/classify_id_ood.py` 刷新两份判定表 → 更新 MANIFEST。

`.gitignore` 按命名模式忽略资产本体与 sidecar，避免 18 GB 误入库：

```
*_????????.glb
*_????????.json
```

不要用裸 `*.json` —— 会连 `road_ood_docum/road_ood_candidates.json` 一起吞掉。
原有 12 个模型是已跟踪文件，不受忽略规则影响；文档与 README 保留可入库。
CC-BY 署名信息在 sidecar 和 `road_ood_manifest.csv` 里各有一份，忽略 sidecar 不会丢署名。
