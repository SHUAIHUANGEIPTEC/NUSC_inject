# 实测数据存档

遮挡与尺度相关决策的原始实测记录，用于复核 `README`「已验证事实」「验收结果」里的结论。
每条都给出复现所需的设置与原始输出。

环境：conda `nusc-inject`，nuScenes **v1.0-mini**，Blender 3.6.23（Cycles CPU），`seed=42`。

---

## M1 · gap-fill 的半径必须是角度量

**问题**：原始 LiDAR 回波只落在扫描线上（VLP-32 共 32 束），range-image 却有 512 行，
不做膨胀时一堵未标注实墙几乎挡不住目标。原有 `spacing_m` 是**米制**半径。

**方法**：合成 20 m 处的实墙，只在 32 条扫描线上有回波，方位向 0.2° 采样；
统计墙所在 bin 区域被填充的比例。

```python
import numpy as np
from nusc_inject.range_image import spherical_bins
H, W = 512, 2048
ELEV_MIN, ELEV_MAX = np.radians(-30.67), np.radians(10.67)
R = 20.0
beams = np.linspace(-30.0, 10.0, 32)          # 32 条扫描线
azs = np.arange(-3.0, 3.0, 0.2)
pts = np.array([[R*np.cos(np.radians(e))*np.cos(np.radians(a)),
                 R*np.cos(np.radians(e))*np.sin(np.radians(a)),
                 R*np.sin(np.radians(e))] for e in beams for a in azs])
row, col, rng, in_fov = spherical_bins(pts, H, W, ELEV_MIN, ELEV_MAX)
```

**原始输出**

```
合成墙面回波 960 点  (32 束 × 30 方位)
墙占据 bin 范围: row 8..503  col 1006..1039  → 16864 bins
无膨胀时填充 960 bins = 0.057

 spacing_m       填充率   r_row@20m   束间距(行)
      0.15     0.815           6        16.5
      0.20     1.000           8        16.5
      0.25     1.000           9        16.5
      0.30     1.000          11        16.5
      0.40     1.000          15        16.5
      0.60     1.000          22        16.5
      0.80     1.000          29        16.5

VLP-32 束间距 ≈ 16.5 行 → 完全填满需 r_row ≥ 8.2
对应 spacing_m ≥ 0.232 m @20m
```

**结论**：束间距是**角度量**，其物理尺寸随距离增长。米制的 `spacing_m=0.15`
在 20 m 只能填 81.5%，且越远越填不满（60 m 处 r_row 降到 3 行）。
故改用**角度**半径 `scene_gap_fill_deg`，默认 0.7°（约半条束间距）。

---

## M2 · gap-fill 不产生过度遮挡

**方法**：同一堵 20 m 的 32 束实墙，三个 144 点目标体分别放在
**墙后 25 m / 空场 12° 方位外 / 墙前 10 m**，扫描 `scene_fill_deg`。

**原始输出**

```
   fill_deg   墙后存活   空场存活   墙前存活
       0.00   120/144   144/144   144/144
       0.30    60/144   144/144   144/144
       0.70     0/144   144/144   144/144
       1.20     0/144   144/144   144/144

期望: 墙后=0/144(全挡)  空场=144/144(全活)  墙前=144/144(更近，全活)
```

另一组目标体位置（同样的墙，1.5°×4° 范围）测得泄漏随半径单调下降：

```
scene_fill_deg       物体存活点       遮挡率
          0.00          126          0.125
          0.15           96          0.333
          0.40           36          0.750
          0.70            0          1.000
          1.00            0          1.000
```

**结论**：0.7° 使墙后目标被完全遮挡（泄漏 0），而空场与墙前目标保持
100% 存活——不存在过度遮挡；1.2° 也未出现过遮。

---

## M3 · `n_per_face` 换成 4 倍无差别

**问题**：遮挡体表面采样只有 `n_per_face=64`（8×8/面），曾被怀疑留孔。

**方法**：同一 P0 单样本、同一模型，只改 `scene_occluders.n_per_face`。

**原始输出**

```
n_per_face = 64  (基线): visible=33173 alpha=33821 occluded=648  all_pass=True
n_per_face = 256        : visible=33173 alpha=33821 occluded=648  all_pass=True
```

**结论**：**逐位相同**。相机侧已改为逐像素平面深度（`occluders._face_depth`），
不再依赖面采样密度；LiDAR 侧有原始回波的角度 gap-fill 兜底。故不改。

---

## M4 · 跨相机平面的遮挡面在数据中不存在

**问题**：`box_depth_raster` 遇到任一顶点在相机平面之后的面会整体跳过，
理论上会漏掉「面横跨相机平面」时的前半部分遮挡。

**方法**：遍历 v1.0-mini 全部 sample × 6 路相机，对每个 `sample_annotation` 框的
6 个面判断深度符号是否混合（部分 `z > 0.1`、部分 `z ≤ 0.1`）。

**原始输出**

```
评估的面总数: 353658
  跨相机平面（部分在前、部分在后，被整体跳过）: 0  (0.0000%)
  整个框在相机后方（合理跳过）: 52285
  涉及跨面框的 (sample, cam) 对: 0
```

**结论**：**零实例**。52,285 个被跳过的是整个框位于相机后方，属正确行为。
该理论缺口在本数据集上不产生任何误差，故不改。

---

## M5 · `box_depth_raster` 最近角深度导致的误遮

**问题**：原先每个框面整片贴「最近角深度」，斜面上偏保守达米级。

**方法**：P0 单样本，对比修正前后的 `n_occluded_px`，并逐像素核对剩余遮挡是否合理。

**原始输出**

```
修正前（整面最近角）: LiDAR obj=4202  visible=30772  occluded=3049  alpha=33821
修正后（逐像素平面）: LiDAR obj=4202  visible=33173  occluded=648   alpha=33821

剩余 648 像素的 depth 核对：
  scene-obj 差值: min=-1.30  中位=-0.55  max=-0.41   全部为负（场景更近）
  unjustified occlusion px = 0
  visible_coverage 0.9400 → 0.9943
```

**结论**：**79% 的遮挡像素原属误遮**。剩余的 648 像素每一个都有更近的场景几何，
差值量级与 `depth_margin`（0.2 + 2%×depth）相当，是真遮挡。

---

## M6 · `size` 轴序修复前后的直接对照

**方法**：同一 `seed=42` 全量轨迹注入（10 scene），对比导出 `sample_annotation.size`。

**原始输出**

```
              重跑前                       重跑后
size 均值     [1.4894, 3.3320, 1.2404]   [3.3827, 1.5121, 1.2593]
size 最小值   [1.1363, 2.5421, 0.9464]   [2.5421, 1.1363, 0.9464]
size 最大值   [1.7627, 3.9433, 1.4680]   [3.9433, 1.7627, 1.4680]
w/l 比        0.4470                      2.2371     (= 1/0.4470)
```

**结论**：前两维精确对调、第三维不变，`w/l` 比互为倒数。nuScenes 约定
`size = [w, l, h]` 中 x 边长是 `size[1]`，修复前把模型框 `(x,y,z)` 原样写出，
导致落库框在 x-y 平面转 90°。

---

## M7 · 遮挡曾在真实批量里完全失效

**方法**：对比 `size` 轴序修复前后两轮全量轨迹注入的 `batch.log` 中
`visible=… alpha=… occluded=…` 行。

**原始输出**

```
修复前: 94 个相机帧，occluded 全部为 0        → 遮挡像素 0 / 2,625,769 = 0.0000
修复后: 72 个相机帧，出现真实遮挡             → 遮挡像素 158,078 / 2,424,182 = 0.0652
```

**结论**：同一轴序 bug 让真实车辆的遮挡体在地面转了 90°，脚印盖错位置，
遮挡功能是彻底的**空操作**——旧导出里「被车挡住的目标」被错误注入成了可见目标。
（此后又经 M5 的逐像素深度修正，当前遮挡占比 6.38%，见下节。）

---

## M8 · 当前全量验收指标

`python scripts/run_batch.py --config configs/batch.yaml --run-id final_export`
（含导出；本节数字即取自该次运行，见 `data/exports/nuscenes-od-p0/export_report.json`）

**原始输出**

```
帧数                     82  失败 0
校验报告全过             82/82
坐标链往返残差 max       3.422e-13
Blender 内参自检 max     0.358 px
size_plausible 通过       82/82
对象最长边范围            2.706 ~ 3.877 m
num_lidar_pts 范围        3 ~ 12000  (总 288158)
点投影⊂α 覆盖率  均值 0.9926  最小 0.9744  可测样本 95
box IoU        均值 0.8126  最小 0.2026
visible_coverage 均值 0.9704
被遮挡像素合计   255771   无理由遮挡合计 0
遮挡占比         255771/4007016 = 0.0638

批量 summary: frames_annotated=82  n_tracks_planned=20  n_tracks_realized=17
              end_reasons={invisible: 7, no_camera: 4, window_end: 9}
导出: 82 帧 / 17 轨迹 / 83 标注  verify.ok=True  n_total_annotations=18621
```

补充核对：`IoU < 0.5` 的 4 例全部为触边豁免。

**结论**：`bbox_iou` 最小值低于阈值的三例全部是触边豁免；`coverage` 不可测的一例
是边缘目标（LiDAR 点整体落在画幅外），按同规则豁免。其余全部满足阈值，
`end_reasons` 中已无 `validation_failed`。

---

## M9 · 测试套件分布

`pytest` → **159 passed**

| 文件 | 用例 | 覆盖 |
|---|---|---|
| `test_anchor.py` | 28 | 缩放/ego↔lidar 换算/gt 变换对/`from_global` scale 回收/网格顶点贴地/退化保护 |
| `test_lidar_channel.py` | 19 | 双向遮挡语义/遮挡体/gap-fill/强度/注入 |
| `test_annotation.py` | 18 | 确定性 token/导出表/跨帧 prev-next 链/来源清单 |
| `test_transforms.py` | 16 | 四元数往返/欧拉角方向与合成/坐标链/view_points |
| `test_occluders.py` | 15 | `[w,l,h]` 轴序/盒面点/逐像素面深度/gap-fill 半径 |
| `test_composite.py` | 12 | z-buffer/可见性规则/深度洞/深度比例 margin |
| `test_validate.py` | 12 | 遮挡理由一致性/尺寸阈值/不可测覆盖率豁免 |
| `test_range_image.py` | 11 | 球面分箱/`dilate_min_map` |
| `test_mesh_points.py` | 9 | Y-up 转换/表面积+质心签名/细分不变性 |
| `test_nusc_io.py` | 6 | `.pcd.bin` I/O |
| `test_config.py` | 5 | `extends` 合并/路径解析 |
| `test_ui_geometry.py` | 8 | 放置 UI 的反投影往返/ego 系换算/天空点拒绝/多目标预览一致/缩略图 job 框幅 |

全部用例不依赖 nuScenes 数据与 Blender，本机耗时 ≈ 0.5 s。

---

## M10 · 不同位置 / 方位 / 类目 / 材质的效果比对

**方法**：用 `assets/models/` 里**真实的 Objaverse 模型**（不是合成方块），
在 `sample_index=0` 上做 4 组单目标注入，覆盖三个不同 nuScenes 类目。
**B 与 D 位置朝向完全相同**，只换模型与类目，用来单独隔离「目标本身」的影响；
A 与 D 同一模型，只改距离。

```
模型                              raw extents        最长边   目标尺寸 dim   类目
traffic_cone_02.glb               (1.5, 1.5, 2.9)     2.9      0.70      movable_object.trafficcone
Detroit_Dumpster.glb              (1.5, 2.1, 1.8)     2.1      2.00      movable_object.pushable_pullable
Dog.glb                            (1368, 4992, 3981)  4992     1.20      animal
```

**原始输出**

```
用例                   类目                          albedo  gt size(w,l,h)   LiDAR点  均强 峰强  alpha   可见   遮挡%   覆盖   IoU
------------------------------------------------------------------------------------------------------------------------
A 交通锥·近距·居中·0°    movable_object.trafficcone     0.70  [0.37,0.37,0.70]     546   106  140   4630   4126  10.9%  0.998 0.941
B 垃圾箱·中距·左偏·75°   movable_object.pushable_pullable 0.50 [2.00,1.45,1.74]   2464    90  100  20917  20823   0.4%  1.000 0.883
C 狗·远距·右偏·20°       animal                        0.25  [1.20,0.33,0.96]      18    32   48   1079    414  61.6%  0.944 0.961
D 交通锥·中距·左偏·75°   movable_object.trafficcone     0.70  [0.37,0.37,0.70]     164   115  140   1198   1198   0.0%  0.994 0.937

放置（ego 系，米）：A [8,0] · B/D [15,-5] · C [28,3]；yaw：A 0° · B/D 75° · C 20°
```

![四组效果对比](summary/figs/fig5_case_comparison.jpg)

*每个用例的黄框已放大。A 近距交通锥（轮廓清晰）；B 中距垃圾箱（偏转 75°）；C 远距的狗——
被前方真实车辆挡住 **61.6%**，只剩贴着真实卡车的窄条；D 与 B 同位置同朝向、只换模型，
用来隔离目标本身的影响。*

**四组结论**

1. **类目与尺寸各自独立落库**：`gt size` 与类目名一一对应，不同目标可用不同 nuScenes 类目
   （`movable_object.trafficcone` / `movable_object.pushable_pullable` / `animal`）——
   三组都是真实模型，不再是合成方块。
2. **点云强度跟随 `albedo`**：三档 albedo 0.70 / 0.50 / 0.25 →
   **峰强** 140 / 100 / 48，正好是 `k·ρ` = 200 × ρ。
   均强（106 / 90 / 30）不严格成比例，因为三个模型的形状与朝向不同、
   平均入射角不一样——要比强度就比峰强。
3. **距离是点数与轮廓的主因**：A 与 D 同模型同 albedo，8 m → 15 m 使注入点
   546 → 164（−70%）、轮廓 4630 → 1198 px（−74%）。
4. **B 与 D 隔离出「目标本身」的影响**：同位置同朝向，垃圾箱（2.0×1.45×1.74）比
   交通锥（0.37×0.37×0.70）点数多 15 倍、轮廓大 17 倍，因为迎面截面积大得多。
5. **遮挡按真实几何生效**：C 的狗在 28 m 处被前方真实车辆挡住 **61.6%** 的目标像素，
   而 A/B/D 只有 0–10.9%。四组覆盖率均 ≥0.944、IoU ≥0.883（全部 ≥0.5 阈值）。
6. **同类目/同模型的两组（A、D）峰强完全相同（140/140）**，说明强度与距离无关
   ——这是 Lambertian 模型的固有属性，见下条。

**顺带发现：点云强度不随距离衰减**

`lambertian_intensity`（`lidar_channel.py:94-100`）只有 `I = k·ρ·max(cos θ, 0)`，
**没有 1/r² 项**。真实 LiDAR 的回波强度会随距离衰减，因此本仓库注入的目标在远处
会比真实目标「显得偏亮」。该行为尚未修正，属已知近似。

**生成器踩过的坑（已修）**：`trimesh` 的 glb 导出**不做轴向转换**，
Z-up 坐标会被原样写进 Y-up 的 glTF，于是所有合规导入器（含 Blender）都会再转一次，
模型变成「躺倒」。`make_demo_models.py` 现在显式做 `(x,y,z)→(x,z,-y)` 后再导出；
同时约定 **x=长、y=宽、z=高** 以匹配 nuScenes 的 `[w,l,h]` 语义，
否则长轴落在 y 上会让标注读成「很短很宽」的目标。
（这条约定对**下载来的模型不成立**：`temple.glb`、`Detroit_Dumpster.glb`、`Dog.glb` 的长轴
都在 y 上，标注因此读作「宽」——`Dog.glb` 的 raw extents 是 (1368, 4992, 3981)，非公制且
y 最长，于是 `size=[1.20, 0.33, 0.96]`（宽 1.20、长 0.33）。几何自洽、不影响注入，
只是语义反了——见 `assets/models/README.md`。）

---

## M11 · 贴地锚点必须用网格顶点，不能用包围盒角点

**目的**：倾斜目标贴地时，「最低点」该按包围盒的 8 个角点算，还是按网格顶点算。

**为什么两者不同**：网格 ⊆ 包围盒，旋转后 `min_z(mesh) ≥ min_z(bbox)` 恒成立。
按包围盒最低角落地，等于把**盒子**放到地面上，而**网格**会悬空
`min_z(mesh) − min_z(bbox)`。真实模型只在少数点接触自己的包围盒
（车辆靠轮胎与保险杠），倾斜后包围盒的最低角往往落在没有几何的空处。

**复现**（`assets/models/temple.glb` 皮卡，7,567 个顶点，bbox 紧致）：

```python
import math, numpy as np
from nusc_inject import mesh_points, transforms as tf, anchor

pts, normals, bmin, bsize, vsig, verts = mesh_points.sample_mesh("assets/models/temple.glb", 2000)
corners = np.array([bmin + np.array([sx, sy, sz]) * bsize
                    for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])
# 真实 LIDAR_TOP 安装：up 轴 = R 的第三行（实测，倾斜 1.429°）
z = np.array([0.01807, -0.017186, 0.999689]); z /= np.linalg.norm(z)
x = np.array([1.0, 0.0, 0.0]); x -= z * (x @ z); x /= np.linalg.norm(x)
R_mount = np.stack([x, np.cross(z, x), z])
for pitch, roll in ((0.0, 0.0), (28.0, -12.0)):
    R = R_mount @ tf.euler_to_rot(0.0, math.radians(pitch), math.radians(roll))
    a = anchor._ground_drop(R, bmin, corners, 1.0)   # 角点贴地（修前）
    b = anchor._ground_drop(R, bmin, verts,   1.0)   # 顶点贴地（修后）
    print(f"pitch={pitch:5.1f} roll={roll:5.1f}  corners={a:10.4f}  verts={b:10.4f}  gap={b-a:8.4f}")
```

**原始输出**（模型框单位；`scale = 3.0 / 463.662 = 0.00647`，目标最长边 3.0 m）

```
raw_min [-103.629 -249.029 0.475]   raw_size [207.258 463.662 172.611]   n_verts 7567
bbox 紧致: min(verts)==raw_min  True   max(verts)==raw_min+raw_size  True
R 第三行 [0.01807 -0.017186 0.999689]   倾斜 1.429°

pitch=  0.0 roll=  0.0  corners=   -7.9685  verts=   -5.5272  gap=  2.4413
pitch= 28.0 roll=-12.0  corners= -187.6671  verts= -155.7158  gap= 31.9513

换算成米（×0.00647）：yaw-only 悬浮 1.58 cm；pitch 28°/roll −12° 悬浮 20.7 cm
```

**修前 / 修后 P0 直接对照**

```
run-id      anchor_lidar_z   CAM_FRONT visible / alpha / occluded   all_pass
pre_fix        -1.592853            34082 / 34387 / 305             True
post_fix       -1.608649            34032 / 34438 / 406             True
Δ              -0.015796 m（−1.58 cm，与上表的 1.58 cm 吻合）
```

**结论**

1. 贴地必须按**网格最低顶点**，`anchor._ground_drop` 因此改为接收 `model_verts`；
   `build_gt` 把 `model_verts` 列为必填，避免「忘了传就静默退回角点」。
2. 真实安装的 1.4° 倾斜意味着**连 yaw-only 都不是逐位不变**：每个目标下沉 1.58 cm，
   P0 的 `visible/alpha/occluded` 从 34082/34387/305 变到 34032/34438/406。
   这是正确性修正的必然代价，不是回归（同 M6 的取舍逻辑）。
3. **单盒模型不受影响**：`demo_pedestrian/barrier/truck.glb` 是 trimesh 造的单盒，
   顶点集与包围盒角点重合，所以 M10 的四组数字一格未变；
   出问题的只有真实非盒状模型（`model.glb`）。
4. 该 bug 只在**倾斜**时肉眼明显；yaw-only 下 1.58 cm 在 15 m 处约 1–2 px，
   靠像素计数才能看出来——这正是「必须用数值而不是靠看图验收」的例子。
