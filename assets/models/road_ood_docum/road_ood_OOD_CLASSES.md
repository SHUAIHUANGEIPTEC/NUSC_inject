# ID / OOD 类目划分（108 类）

判定口径见 `scripts/classify_id_ood.py` 头部注释。依据：本地 nuScenes v1.0-mini 的
`category.json`（23 类全税则）+ devkit `DETECTION_NAMES`（10 类评测口径）。

| 档位 | 类目 | 模型 | 占比 | 体积 |
|---|---|---|---|---|
| `ID-CORE` ID 主集（detection 10 类） | 8 | 244 | 7.0% | 1.49 GB |
| `ID-TAIL` ID 长尾（23 类税则内） | 47 | 1570 | 45.0% | 7.27 GB |
| `OOD` OOD（税则外） | 53 | 1672 | 48.0% | 9.99 GB |
| **合计** | **108** | **3486** | 100% | **18.75 GB** |

## ID-CORE — ID 主集（detection 10 类）（8 类 / 244 个）

| LVIS 类目 | 中文 | 分组 | 模型数 | 建议注入 category | 依据 |
|---|---|---|---|---|---|
| `bicycle` | 自行车 | 非典型交通参与者 | 44 | `vehicle.bicycle` | nuScenes 有 vehicle.bicycle，属 detection 10 类 |
| `motorcycle` | 摩托车 | 非典型交通参与者 | 37 | `vehicle.motorcycle` | nuScenes 有 vehicle.motorcycle，属 detection 10 类 |
| `bulldozer` | 推土机 | 施工/工业设备 | 20 | `vehicle.construction` | nuScenes vehicle.construction 的典型实例 |
| `forklift` | 叉车 | 施工/工业设备 | 28 | `vehicle.construction` | nuScenes vehicle.construction 定义覆盖 |
| `power_shovel` | 挖掘机 | 施工/工业设备 | 13 | `vehicle.construction` | 挖掘机 = vehicle.construction 的典型实例 |
| `sawhorse` | 锯木架/施工支架 | 遗撒物/杂物 | 9 | `movable_object.barrier` | 施工用锯木架即临时路障支架，归 barrier |
| `cone` | 交通锥 | 道路设施 | 80 | `movable_object.trafficcone` | nuScenes 有 movable_object.trafficcone，且属 detection 10 类 |
| `jersey` | 泽西护栏/水泥隔离墩 | 道路设施 | 13 | `movable_object.barrier` | 泽西护栏 = nuScenes movable_object.barrier 的典型形态 |

## ID-TAIL — ID 长尾（23 类税则内）（47 类 / 1570 个）

| LVIS 类目 | 中文 | 分组 | 模型数 | 建议注入 category | 依据 |
|---|---|---|---|---|---|
| `bear` | 熊 | 动物 | 25 | `animal` | nuScenes 确有 animal 类目，但全库仅个位数实例 |
| `camel` | 骆驼 | 动物 | 12 | `animal` | 同上：animal 税则内，极稀有 |
| `cat` | 猫 | 动物 | 45 | `animal` | 同上：animal 税则内，极稀有 |
| `chicken_(animal)` | 鸡 | 动物 | 29 | `animal` | 同上：animal 税则内，极稀有 |
| `cow` | 母牛 | 动物 | 30 | `animal` | 同上：animal 税则内，极稀有 |
| `deer` | 鹿 | 动物 | 25 | `animal` | 同上：animal 税则内，极稀有 |
| `dog` | 狗 | 动物 | 52 | `animal` | 同上：animal 税则内，极稀有 |
| `domestic_ass` | 驴 | 动物 | 7 | `animal` | 同上：animal 税则内，极稀有 |
| `duck` | 鸭 | 动物 | 18 | `animal` | 同上：animal 税则内，极稀有 |
| `elephant` | 大象 | 动物 | 50 | `animal` | 同上：animal 税则内，极稀有 |
| `frog` | 青蛙 | 动物 | 57 | `animal` | 同上：尺寸过小，animal 税则内 |
| `giraffe` | 长颈鹿 | 动物 | 39 | `animal` | 同上：animal 税则内，极稀有 |
| `goat` | 山羊 | 动物 | 8 | `animal` | 同上：animal 税则内，极稀有 |
| `goose` | 鹅 | 动物 | 17 | `animal` | 同上：animal 税则内，极稀有 |
| `hippopotamus` | 河马 | 动物 | 15 | `animal` | 同上：animal 税则内，极稀有 |
| `hog` | 猪 | 动物 | 41 | `animal` | 同上：animal 税则内，极稀有 |
| `horse` | 马 | 动物 | 34 | `animal` | 同上：animal 税则内，极稀有 |
| `lion` | 狮子 | 动物 | 60 | `animal` | 同上：animal 税则内，极稀有 |
| `monkey` | 猴子 | 动物 | 56 | `animal` | 同上：animal 税则内，极稀有 |
| `rabbit` | 兔子 | 动物 | 46 | `animal` | 同上：animal 税则内，极稀有 |
| `rhinoceros` | 犀牛 | 动物 | 38 | `animal` | 同上：animal 税则内，极稀有 |
| `sheep` | 绵羊 | 动物 | 14 | `animal` | 同上：animal 税则内，极稀有 |
| `snake` | 蛇 | 动物 | 48 | `animal` | 同上：尺寸过小，animal 税则内 |
| `squirrel` | 松鼠 | 动物 | 20 | `animal` | 同上：尺寸过小，animal 税则内 |
| `tiger` | 老虎 | 动物 | 39 | `animal` | 同上：animal 税则内，极稀有 |
| `turtle` | 乌龟 | 动物 | 51 | `animal` | 同上：尺寸过小，animal 税则内 |
| `zebra` | 斑马 | 动物 | 14 | `animal` | 同上：animal 税则内，极稀有 |
| `baby_buggy` | 婴儿车 | 非典型交通参与者 | 7 | `human.pedestrian.stroller` | nuScenes 有 human.pedestrian.stroller，mini 零标注 |
| `motor_scooter` | 电动踏板车 | 非典型交通参与者 | 37 | `vehicle.motorcycle` | 电动踏板/小排量踏板车，归 motorcycle 但形态不标准 |
| `shopping_cart` | 购物车 | 非典型交通参与者 | 14 | `movable_object.pushable_pullable` | 推车 = pushable_pullable 的典型例子 |
| `skateboard` | 滑板 | 非典型交通参与者 | 84 | `human.pedestrian.personal_mobility` | nuScenes 有 human.pedestrian.personal_mobility，mini 25 帧 |
| `wheelchair` | 轮椅 | 非典型交通参与者 | 25 | `human.pedestrian.wheelchair` | nuScenes 有 human.pedestrian.wheelchair，mini 零标注 |
| `barrel` | 桶 | 遗撒物/杂物 | 63 | `movable_object.debris` | 油桶：可移动、留在可行驶面上 —— debris 的定义 |
| `barrow` | 独轮手推车 | 遗撒物/杂物 | 14 | `movable_object.pushable_pullable` | 独轮手推车 = 推/拉对象 |
| `box` | 盒子 | 遗撒物/杂物 | 36 | `movable_object.debris` | 纸箱遗撒，debris 定义内 |
| `bucket` | 桶 | 遗撒物/杂物 | 44 | `movable_object.debris` | 桶类遗撒，debris 定义内 |
| `cooler_(for_food)` | 保鲜箱（用于食物） | 遗撒物/杂物 | 7 | `movable_object.pushable_pullable` | 保温箱：可推拉的小件 |
| `crate` | 板条箱 | 遗撒物/杂物 | 88 | `movable_object.debris` | 板条箱遗撒，debris 定义内 |
| `handcart` | 手推车 | 遗撒物/杂物 | 31 | `movable_object.pushable_pullable` | 手推车 = 推/拉对象 |
| `ladder` | 梯子 | 遗撒物/杂物 | 18 | `movable_object.debris` | 梯子掉落，debris 定义内 |
| `log` | 原木 | 遗撒物/杂物 | 47 | `movable_object.debris` | 原木掉落，debris 定义内 |
| `mattress` | 床垫 | 遗撒物/杂物 | 22 | `movable_object.debris` | 床垫遗撒，debris 定义内 |
| `pipe` | 管材 | 遗撒物/杂物 | 19 | `movable_object.debris` | 管材遗撒，debris 定义内 |
| `suitcase` | 手提箱 | 遗撒物/杂物 | 27 | `movable_object.pushable_pullable` | 行李箱是 nuScenes pushable_pullable 的教科书例子 |
| `tarp` | 防水帆布 | 遗撒物/杂物 | 8 | `movable_object.debris` | 篷布遗撒（可覆盖路面），debris 定义内 |
| `dumpster` | 垃圾箱 | 道路设施 | 46 | `movable_object.pushable_pullable` | 大型垃圾箱：项目已实测用 pushable_pullable 落库；该类在 mini 仅 82 帧 |
| `trash_can` | 垃圾桶 | 道路设施 | 43 | `movable_object.pushable_pullable` | 垃圾桶归 pushable_pullable；mini 仅 82 帧（全类合计） |

## OOD — OOD（税则外）（53 类 / 1672 个）

| LVIS 类目 | 中文 | 分组 | 模型数 | 建议注入 category | 依据 |
|---|---|---|---|---|---|
| `boat` | 船 | 非典型交通参与者 | 30 | `movable_object.anomaly` | 船舶，nuScenes 城市场景无此税则 |
| `canoe` | 独木舟 | 非典型交通参与者 | 22 | `movable_object.anomaly` | 独木舟，非道路目标 |
| `golfcart` | 高尔夫球车 | 非典型交通参与者 | 12 | `movable_object.anomaly` | 场地车，nuScenes 无对应税则 |
| `horse_buggy` | 马车 | 非典型交通参与者 | 7 | `movable_object.anomaly` | 马车，nuScenes 无对应税则 |
| `horse_carriage` | 马车 | 非典型交通参与者 | 14 | `movable_object.anomaly` | 马车，nuScenes 无对应税则 |
| `kayak` | 皮艇 | 非典型交通参与者 | 21 | `movable_object.anomaly` | 皮划艇，非道路目标 |
| `sled` | 雪橇 | 非典型交通参与者 | 6 | `movable_object.anomaly` | 雪橇，非道路目标 |
| `snowmobile` | 雪地摩托 | 非典型交通参与者 | 7 | `movable_object.anomaly` | 雪地摩托，nuScenes 无对应税则 |
| `surfboard` | 冲浪板 | 非典型交通参与者 | 19 | `movable_object.anomaly` | 冲浪板，非道路目标 |
| `generator` | 发电机 | 施工/工业设备 | 35 | `movable_object.anomaly` | 发电机组：非车辆、非路面可移动物 |
| `lawn_mower` | 割草机 | 施工/工业设备 | 10 | `movable_object.anomaly` | 割草机：可移动但 nuScenes 无此税则 |
| `plow_(farm_equipment)` | 犁（农用设备） | 施工/工业设备 | 14 | `movable_object.anomaly` | 农具：nuScenes construction 限定为施工用途 |
| `tractor_(farm_equipment)` | 拖拉机 | 施工/工业设备 | 40 | `movable_object.anomaly` | 农用拖拉机：nuScenes construction 限定为施工用途 |
| `armoire` | 衣柜 | 遗撒物/杂物 | 52 | `movable_object.anomaly` | 整体衣柜，体量与路面遗撒物不符 |
| `automatic_washer` | 自动洗衣机 | 遗撒物/杂物 | 22 | `movable_object.anomaly` | 家电，nuScenes 无对应税则 |
| `bathtub` | 浴缸 | 遗撒物/杂物 | 14 | `movable_object.anomaly` | 卫浴大件，nuScenes 无对应税则 |
| `bed` | 床 | 遗撒物/杂物 | 33 | `movable_object.anomaly` | 大件家具 |
| `bookcase` | 书柜 | 遗撒物/杂物 | 58 | `movable_object.anomaly` | 大件家具 |
| `cabinet` | 柜子 | 遗撒物/杂物 | 44 | `movable_object.anomaly` | 大件家具 |
| `plastic_bag` | 塑料袋 | 遗撒物/杂物 | 7 | `movable_object.anomaly` | 尺寸过小，不足以构成可检测立体目标 |
| `refrigerator` | 冰箱 | 遗撒物/杂物 | 28 | `movable_object.anomaly` | 家电，nuScenes 无对应税则 |
| `sink` | 水槽 | 遗撒物/杂物 | 7 | `movable_object.anomaly` | 卫浴件 |
| `sofa` | 沙发 | 遗撒物/杂物 | 45 | `movable_object.anomaly` | 大件家具 |
| `table` | 桌子 | 遗撒物/杂物 | 30 | `movable_object.anomaly` | 大件家具 |
| `toilet` | 马桶 | 遗撒物/杂物 | 21 | `movable_object.anomaly` | 卫浴件 |
| `awning` | 遮阳篷 | 道路设施 | 26 | `movable_object.anomaly` | 路边固定设施，nuScenes 不标注静态路面附属物 |
| `bench` | 长凳 | 道路设施 | 49 | `movable_object.anomaly` | 路边固定设施，非交通参与者 |
| `billboard` | 广告牌 | 道路设施 | 22 | `movable_object.anomaly` | 固定广告牌，nuScenes 不标注 |
| `chair` | 椅子 | 道路设施 | 80 | `movable_object.anomaly` | 家具类固定物 |
| `fire_alarm` | 火警报警器 | 道路设施 | 9 | `movable_object.anomaly` | 建筑附属，尺寸过小 |
| `fire_extinguisher` | 灭火器 | 道路设施 | 83 | `movable_object.anomaly` | 建筑附属，尺寸过小 |
| `fireplug` | 消防栓 | 道路设施 | 93 | `movable_object.anomaly` | 消防栓：KITTI-OoD 常见，但 nuScenes 无此税则 |
| `flagpole` | 旗杆 | 道路设施 | 21 | `movable_object.anomaly` | 杆状固定设施 |
| `garbage` | 垃圾/垃圾袋 | 道路设施 | 29 | `movable_object.anomaly` | 散落垃圾，尺寸远小于 debris 的下限 |
| `lamppost` | 路灯 | 道路设施 | 54 | `movable_object.anomaly` | 杆状固定设施（nuScenes 只标可移动物/车/人） |
| `mail_slot` | 信箱 | 道路设施 | 17 | `movable_object.anomaly` | 建筑附属，尺寸过小 |
| `mailbox_(at_home)` | 家用邮箱 | 道路设施 | 25 | `movable_object.anomaly` | 私人信箱，尺寸过小 |
| `manhole` | 下水道井盖 | 道路设施 | 82 | `movable_object.anomaly` | 嵌于路面，无法作为独立立体目标注入 |
| `milestone` | 里程碑 | 道路设施 | 19 | `movable_object.anomaly` | 路侧小构件 |
| `newsstand` | 报摊 | 道路设施 | 38 | `movable_object.anomaly` | 固定售卖亭 |
| `parking_meter` | 停车计时器 | 道路设施 | 15 | `movable_object.anomaly` | 路侧小构件 |
| `pole` | 杆 | 道路设施 | 38 | `movable_object.anomaly` | 杆状固定设施 |
| `postbox_(public)` | 邮箱（公共） | 道路设施 | 32 | `movable_object.anomaly` | 公共邮筒，nuScenes 不标注 |
| `signboard` | 标牌 | 道路设施 | 49 | `movable_object.anomaly` | 店招/标牌，非交通标志 |
| `silo` | 筒仓 | 道路设施 | 16 | `movable_object.anomaly` | 工业建筑体 |
| `stop_sign` | 停车标志 | 道路设施 | 29 | `movable_object.anomaly` | 停车标志：KITTI-OoD 高频，nuScenes 明确不标交通标志 |
| `street_sign` | 街道标志 | 道路设施 | 36 | `movable_object.anomaly` | 路牌，nuScenes 不标注 |
| `streetlight` | 路灯 | 道路设施 | 22 | `movable_object.anomaly` | 杆状固定设施 |
| `telephone_booth` | 电话亭 | 道路设施 | 47 | `movable_object.anomaly` | 固定亭体 |
| `telephone_pole` | 电话杆 | 道路设施 | 23 | `movable_object.anomaly` | 杆状固定设施 |
| `traffic_light` | 交通信号灯 | 道路设施 | 41 | `movable_object.anomaly` | 信号灯：nuScenes 不标注（依赖地图/先验） |
| `vending_machine` | 自动售货机 | 道路设施 | 64 | `movable_object.anomaly` | 固定设备 |
| `water_tower` | 水塔 | 道路设施 | 15 | `movable_object.anomaly` | 工业建筑体 |
