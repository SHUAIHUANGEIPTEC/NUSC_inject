# assets/models — 入库的第三方 3D 模型

本目录是**入库的输入资产**（与 `data/` 相对：`data/` 全是运行期产物、随时可删）。
新克隆仓库后**直接就能注入**，不需要先跑任何生成脚本。

共 12 个 `.glb`，约 38 MB。每个模型旁边有一份同名 `.json` sidecar，
记录来源与许可（`uid` / LVIS 类目 / Sketchfab URI / 许可 / 顶点数）。

## 许可一览

| 文件 | 体积 | LVIS 类目 | 许可 | 原始名称 |
|---|---|---|---|---|
| `traffic_cone_02.glb` | 2.90 MB | `cone` | CC-BY 4.0 | traffic cone 02 |
| `Modelo_Blanes.glb` | 0.87 MB | `jersey` | CC-BY 4.0 | Modelo Blanes（水泥护栏） |
| `Trashcan_Low.glb` | 4.79 MB | `trash_can` | CC-BY 4.0 | Trashcan Low |
| `Detroit_Dumpster.glb` | 8.31 MB | `dumpster` | CC-BY 4.0 | Detroit Dumpster |
| `Rusty_Stop_Sign.glb` | 7.18 MB | `stop_sign` | CC-BY 4.0 | Rusty Stop Sign |
| `Reb_Road_Street_Sign.glb` | 0.53 MB | `street_sign` | CC-BY 4.0 | Reb Road Street Sign |
| `pole_electric.glb` | 2.18 MB | `telephone_pole` | CC-BY 4.0 | pole electric |
| `British_Postbox.glb` | 1.93 MB | `postbox_(public)` | CC-BY 4.0 | British Postbox |
| `Dutch_postal_services_Mailbox.glb` | 7.15 MB | `mailbox_(at_home)` | CC-BY 4.0 | Dutch postal services Mailbox |
| `Cat.glb` | 0.53 MB | `cat` | CC-BY 4.0 | Cat |
| `DOG.glb` | 1.47 MB | `dog` | CC-BY 4.0 | DOG |
| `temple.glb` | 0.28 MB | — | **unknown** | temple.glb（见下） |

前 11 个来自 **Objaverse v1**（Sketchfab 资产），逐个从
`metadata/{分片}.json.gz` 查过 `license` 字段，只挑了 `by`（CC-BY 4.0）；
`by-nc` / `by-nc-sa` / `by-sa` 的候选一律没要。

Objaverse 数据集整体按 **ODC-By v1.0** 授权，逐对象许可分布见其数据集卡片
（CC-BY 721K / CC-BY-NC 25K / CC-BY-NC-SA 52K / CC-BY-SA 16K / CC0 3.5K）。
引用请用：

```bibtex
@article{objaverse,
  title={Objaverse: A Universe of Annotated 3D Objects},
  author={Matt Deitke and Stefan Schwenk and Jordi Salvador and Luca Weihs and
          Oscar Michel and Elias VanderBilt and Ludwig Schmidt and
          Kiana Ehsani and Aniruddha Kembhavi and Ali Farhadi and Roei Herzig},
  journal={arXiv:2212.08051}, year={2022}
}
```

### `temple.glb`：许可不明

文件里**没有任何许可/署名元数据**（glTF 只有 `asset.generator: Khronos glTF
Blender I/O`）。它来自外部项目 `CARLA-OoD3D`（`04-blenders/temple.glb`），而该项目
的流程是「用 LVIS 类目筛选 Objaverse uid 再下载」，所以它**很可能是 Objaverse 资产**，
但原始 uid 与许可没有记录下来。

本仓库按原样再分发。如果你需要干净的许可链，用
[`scripts/download_models.py`](../../scripts/download_models.py) 自己拉一个替换掉即可。

## 使用注意

- **非公制单位**：`Cat.glb` 的原始 bbox 是 `279×1409×767`、
  `Dutch_postal_services_Mailbox.glb` 是 `5.5×9.8×16`——模型没有统一米制。
  管线靠 `model.target_max_dim_m` 归一（`scale = 最长边 / 模型最长边`），
  注入没问题，只是 `scale` 会很小。
- **长轴方向**：本仓库约定 **x = 长度、y = 宽度、z = 高度**（匹配 nuScenes
  `size = [w, l, h]` 中 `l` 沿 box-x）。`Cat.glb` 的长轴在 y 上，它的标注会读作
  「宽 3 m、长 1.4 m」——几何自洽但语义反了。新拉模型时留意这一点，
  必要时在加载时绕 z 转 90°。
- **`data/models/`** 仍然只放**生成物**（`scripts/make_demo_models.py` 的演示盒子、
  `scripts/make_doc_fixtures.py` 的轴向对照模型），不入库。

## 再拉几个模型

```bash
pip install objaverse          # 可选依赖，装完才有 load_objects()
python scripts/download_models.py --uids <uid1> <uid2> --out assets/models
```

`--uids` 需要 Objaverse 的 uid（可从 LVIS 类目清单筛，见
`scripts/download_models.py` 的 `--uid-file` 用法）。
