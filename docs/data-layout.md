# data/ 目录：内容、用途与产生者

`data/` 下**全部是运行期数据**，不进版本控制（`.gitignore` 已排除）。克隆仓库后该目录不存在：
要么按 §1 自行准备输入，要么由管线生成。

本机当前状态（供参照，数字会随运行增减）：

```
data/                 334M
├── nuscenes          0      → 软链到仓库外部的 nuScenes 数据集
├── models/          20K     第三方模型
├── outputs/        216M     运行产物（8 个目录）
└── exports/        119M     nuScenes 格式导出
```

---

## 1. 目录职责一览

| 路径 | 内容 | 谁产生 | 入库 | 可删 |
|---|---|---|---|---|
| `nuscenes/` | nuScenes 原始数据集（`v1.0-mini/` 13 张表 + `samples/` + `sweeps/` + `maps/`） | **你提供**（官网下载，CC BY-NC-SA 4.0） | 否 | 否，删了跑不了 |
| `models/` | 第三方 3D 模型（`.glb/.gltf/.obj/.fbx/.ply`） | **你提供**；或用 `scripts/make_demo_models.py` 生成演示模型 | 否 | 否 |
| `outputs/p0/<run_id>/` | 单次运行的全部中间产物与校验报告 | `run_p0.py` / `run_batch.py` | 否 | 可，重跑即得 |
| `exports/<name>/` | 标准 nuScenes 格式数据集，devkit 可直接加载 | `run_batch.py` 的导出阶段 | 否 | 谨慎，见 §5 |

`nuscenes/` 建议用软链指到仓库外（数据集很大且不该混进工程目录）：

```bash
ln -s /path/to/nuScenes data/nuscenes
```

---

## 2. 数据流

```
nuScenes 原始数据 ─┐
                   ├─→ Step1 锚点（gt.json）─→ Step2 LiDAR ─┐
第三方模型 ────────┘                    └─→ Step3 相机  ─┤
                                                          ├─→ Step4 合成 ─→ Step6 校验
                                                          │                   ↓
                                                          └─────────────→ Step5 导出 specs.json
                                                                                    ↓
                                                                    data/exports/<name>/
```

`data/outputs/` 是**过程**，`data/exports/` 是**产物**。前者可随时丢弃，后者是要交付的数据集。

---

## 3. 一次运行产物（`data/outputs/p0/<run_id>/`）

单目标运行把文件平铺在 run 目录下；批量与轨迹模式每帧一层：

```
data/outputs/p0/<run_id>/
├── gt.json                     单目标
├── sample_0000/                批量（legacy）
│   └── …（下表文件）
├── scene-0061/f018/            轨迹模式
│   └── …（下表文件）
├── specs.json                  所有帧的 Step-5 导出规格（批量/轨迹模式）
└── batch.log                   批量运行日志
```

单帧目录内：

| 文件 | 内容 | 典型体积 | 用于 |
|---|---|---|---|
| `gt.json` | 单一 3D 真值：`translation/size/rotation` + `T_model2global` / `T_model2lidar` / `T_lidar2global` | 4K | 两通道的唯一输入 |
| `lidar/injected.pcd.bin` | 注入后的点云，原生 (N,5) | 0.7 MB | **交付物**：被导出的点云 |
| `lidar/obj_points_lidar.npy` | 存活的物体点 (M,5) | 小 | 校验与可视化 |
| `lidar/obj_mask.npy` / `obj_ids.npy` | 哪些行是注入点、属于第几个目标 | 小 | 校验与可视化 |
| `blender/CAM_*.png` | Blender 渲染的 RGBA 前景 | 0.6 MB/路 | Step 4 的输入 |
| `blender/result.json` | 每路相机的内参自检误差与遮挡体 bbox 对账 | 小 | 漂移守卫 |
| `composite/CAM_*.jpg` | 合成后的图像 | 0.2 MB/路 | **交付物**：被导出的图像 |
| `depth/CAM_*_obj_depth.npy` | 物体深度缓冲 | **11 MB/路** | 仅调试，可删 |
| `depth/CAM_*_scene_depth.npy` | 场景深度缓冲 | **11 MB/路** | 仅调试，可删 |
| `viz/CAM_*_overlay.jpg` | 注入点投影回图像 | 0.2 MB/路 | 观察一致性 |
| `viz/lidar_bev.png` | LiDAR 俯视图，目标标红 | 0.5 MB | 观察位置 |
| `validation_report.json` | Step-6 五项校验明细 | 8K | **判断该帧是否合格** |
| `job.json` | 发给 Blender 的完整任务（含相机内外参） | 小 | 复现/排查 |
| `spec.json` | 该帧的 Step-5 导出规格（标注行 + 替换文件映射） | 小 | 导出输入 |
| `logs/run.log` / `logs/blender_render.log` | 运行与渲染日志 | 小 | 排查失败 |

> **`depth/*.npy` 是体积大头**：每路相机 22 MB，6 路全渲染就是 132 MB/帧。
> 它们只用于事后分析遮挡判定，删除不影响任何交付物。

---

## 4. 按通道看产物的归属

| 通道 | 产生 | 最终去向 |
|---|---|---|
| LiDAR | `lidar/injected.pcd.bin` | 实拷进 `exports/…/samples/LIDAR_TOP/` |
| 相机 | `composite/CAM_*.jpg` | 实拷进 `exports/…/samples/CAM_*/` |
| 标注 | `spec.json` → `specs.json` | 汇总成 `exports/…/v1.0-mini/sample_annotation.json` 等四张表 |

---

## 5. 导出目录（`data/exports/<name>/`）

```
nuscenes-od-p0/
├── v1.0-mini/          35M   13 张表 = 原表 + 追加的 category/attribute/instance/sample_annotation
├── samples/            84M   4723 个软链（原文件）+ 125 个实拷（被注入的文件）
├── sweeps/            软链    未改动，指向原数据集
├── maps/              软链    同上
├── export_report.json  8K    导出统计 + devkit 校验结果
└── specs.json          92K   逐帧逐目标的来源清单（位姿/尺寸/类目/模型路径/轨道 key）
```

**软链策略**：`samples/` 里绝大部分是指回原数据集的软链，只有**被注入替换过的**文件
（点云 + 合成图，共 125 个）是真实拷贝。所以导出目录只有 84M 却能独立被 devkit 加载——
前提是**原数据集不能删**，软链依赖它。

> ⚠️ `export_dataset` 开头会 `rmtree(export_root)` 重建整个导出目录。测试导出请用
> 独立的 `export_dir`，否则会摧毁成品。

**`specs.json` 是数据集的来源凭证**：它能重建导出（`--export-only`），也是"这 66 个目标
分别是什么、放在哪"的唯一记录。建议随数据集一起保留。

---

## 6. 清理建议

| 想省空间 | 做法 | 省下 |
|---|---|---|
| 只要交付物 | 删 `data/outputs/p0/*/depth/` | 每帧约 22 MB × 相机数 |
| 只保留合格帧 | 按 `validation_report.json` 的 `all_pass` 筛 | 视失败率 |
| 重导出前腾空间 | 保留 `specs.json`，删其余 | 大幅 |
| 换机器/归档 | 只需 `data/exports/<name>/` + `specs.json`（原数据集除外） | — |

**不要删**：`data/nuscenes/`（导出目录的软链依赖它）、`data/exports/<name>/specs.json`。
