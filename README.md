# write_black_smoke — 黑烟白烟火焰检测

基于 YOLOv8 的黑烟/白烟/火焰检测项目，包含数据转换、训练、推理和后处理滤波模块。

## 项目结构

```
├── convert_voc_to_yolo.py    # VOC → YOLO格式转换
├── train.py                  # 训练脚本
├── infer.py                  # 推理/跟踪脚本（集成滤波）
├── val.py                    # 验证评估脚本
├── smoke_filter.py           # 规则过滤 + 时序稳定模块
├── yolo_dataset/             # YOLO格式数据集 (gitignored)
└── runs/                     # 训练/验证输出 (gitignored)
```

## 快速使用

```bash
# 训练
python train.py

# 推理 (默认启用滤波)
python infer.py video.mp4
python infer.py image.jpg
python infer.py 0                     # 摄像头

# 跟踪
python infer.py --track video.mp4

# 关闭后处理（原始检测效果对比）
python infer.py --no-filter video.mp4

# 验证
python val.py
```

---

## 后处理滤波模块 (`smoke_filter.py`)

针对烟/火检测的三大痛点设计：**房屋误检为烟**、**检测框抖动**、**检测不稳定**。

### 一、`RuleFilter` — 规则过滤（去"假烟"）

逐帧过滤不合理检测，采用空间/几何规则，解决房屋→whitesmoke 误检。

#### 过滤规则

| 规则 | 策略 | 目标 |
|------|------|------|
| **宽高比过滤** | blacksmoke/whitesmoke/smoke 的宽高比 > 2.2 且置信度 < 0.6 时过滤；宽高比 > 3.0 直接过滤 | 烟呈柱状/羽状，建筑/墙壁呈扁矩形 |
| **地面建筑过滤** | whitesmoke 宽高比 > 1.8 且框底部在画面底部 75% 以下且置信度 < 0.55 时过滤 | 建筑紧贴地面，烟悬浮在空中 |
| **面积过滤** | 框占图面积 > 85% 或 < 0.2%（且低置信度）时过滤 | 排除铺满全图或微不可见的误检 |
| **边缘位置过滤** | 框顶部/底部紧贴图片边缘（< 5px）时过滤 | 无效的边缘检测 |
| **火焰尺寸约束** | fire 宽高比 > 3.0 或面积 > 50% 时过滤 | 火焰不会过于扁长或过大 |

#### 用法

```python
from smoke_filter import RuleFilter

rf = RuleFilter(img_h=1080, img_w=1920)
filtered_boxes = rf.filter_boxes(raw_boxes)
# raw_boxes: [(x1, y1, x2, y2, conf, cls_id), ...]
```

---

### 二、`TemporalStabilizer` — 时序稳定（去瞬时误检、防闪烁、提高报警可靠性）

跨帧稳定模块，解决检测框抖动、闪烁消失、单帧误检三大问题。

#### 核心设计

| 机制 | 参数 | 说明 |
|------|------|------|
| **置信度迟滞** | `appear_thresh=0.35` > `disappear_thresh=0.20` | 出现门槛高、消失门槛低，防止阈值边界处频繁闪烁 |
| **预热期** | `min_hits=3` | 新检测必须连续命中 3 帧且置信度达标才输出，滤除单帧瞬时误检 |
| **EMA 坐标平滑** | `alpha_box=0.35` | 当前帧与历史轨迹加权平均，消除坐标突变 |
| **EMA 置信度平滑** | `alpha_conf=0.25` | 抑制置信度突变导致的消失/出现 |
| **缺失帧指数衰减** | 每帧 ×0.7 | 轨迹丢失时置信度指数衰减，而非硬性计次，平滑过渡 |
| **逐轨迹闪烁检测** | `flicker_window=15`, 翻转 ≥ 4 次则压制 | 独立跟踪每条轨迹的 visible/hidden 翻转次数，高频闪烁轨迹被压制 |
| **可靠性评分** | 输出 `reliability ∈ [0,1]` | 综合年龄、命中率、置信度稳定性、缺失惩罚，供报警系统使用 |

#### 工作流程

```
当前帧检测 → 贪心匹配(高置信度优先) → 匹配成功: EMA更新坐标/置信度
                                    → 匹配失败: 置信度衰减 ×0.7
新检测 → 置信度≥0.5×appear_thresh → 创建新轨迹(预热期)
已确认轨迹缺失 → 置信度衰减 → 低于 disappear_thresh 则消亡
逐轨迹闪烁检测 → 翻转次数超标 → 压制输出
输出 → (x1,y1,x2,y2,conf,cls_id,reliability)
```

#### 报警可靠性评分

每条输出轨迹附带 `reliability` 评分，由四项因子加权：

| 因子 | 权重 | 说明 |
|------|------|------|
| age_factor | 35% | 累计命中帧数越多越可靠，5帧后趋于饱和 |
| hit_rate_factor | 25% | 命中率越高越可靠 |
| conf_stability | 25% | 当前置信度与峰值置信度的比值，波动小则可靠 |
| miss_penalty | 15% | 缺失帧数惩罚，每缺失一帧扣 10% |

#### 用法

```python
from smoke_filter import TemporalStabilizer

stabilizer = TemporalStabilizer(
    appear_thresh=0.35,      # 出现阈值
    disappear_thresh=0.20,   # 消失阈值（迟滞防闪烁）
    min_hits=3,              # 预热期帧数
)

for frame in video_frames:
    raw_boxes = model(frame)
    stable_boxes = stabilizer.update(raw_boxes)
    # stable_boxes: [(x1,y1,x2,y2,conf,cls_id,reliability), ...]
```

---

### 三、在 `infer.py` 中的集成

两个模块默认在**视频和摄像头模式**下同时开启，处理流程：

```
原始检测 → 规则过滤(RuleFilter) → 时序稳定(TemporalStabilizer) → 可视化输出
```

图片模式下仅启用规则过滤（单帧无需时序稳定）。

使用 `--no-filter` 参数可关闭两个模块，便于对比效果。
使用 `--show-reliability` 参数可在框下方显示可靠性评分。

```bash
python infer.py video.mp4                    # 默认启用滤波
python infer.py --no-filter video.mp4        # 对比原始效果
python infer.py --show-reliability video.mp4 # 显示可靠性评分
```

## 数据集

类别映射 (YOLO ID):

| ID | 类别 | 说明 |
|----|------|------|
| 0 | blacksmoke | 黑烟 |
| 1 | whitesmoke | 白烟 |
| 2 | fire | 火焰 |
| 3 | smoke | 烟 |
