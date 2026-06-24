"""
烟/火检测的后处理滤波模块

包含:
  - RuleFilter:  单帧规则过滤（宽高比、位置、大小），消除建筑误检为whitesmoke
  - TemporalStabilizer:  时序稳定（EMA平滑、框抖动抑制、缺失帧恢复、闪烁抑制）
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  规则过滤模块  (去"假烟")
# ─────────────────────────────────────────────

class RuleFilter:
    """
    基于空间/几何规则的逐帧过滤。
    解决: 房屋→whitesmoke 误检，不合理尺寸/位置的检测。
    """

    def __init__(self, img_h: int, img_w: int):
        self.h = img_h
        self.w = img_w

    def filter_boxes(self, boxes_data: list) -> list:
        """
        输入: boxes_data = [(x1, y1, x2, y2, conf, cls_id), ...]
        输出: 过滤后的列表
        """
        kept = []
        for x1, y1, x2, y2, conf, cls_id in boxes_data:
            if cls_id not in (0, 1, 2, 3):
                kept.append((x1, y1, x2, y2, conf, cls_id))
                continue
            if self._pass_rules(x1, y1, x2, y2, conf, cls_id):
                kept.append((x1, y1, x2, y2, conf, cls_id))
        return kept

    def _pass_rules(self, x1, y1, x2, y2, conf, cls_id) -> bool:
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            return False

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        aspect = box_w / max(box_h, 1)          # 宽高比（>1 扁宽, <1 瘦高）
        area_ratio = (box_w * box_h) / (self.w * self.h)
        bottom_ratio = y2 / self.h               # 框底部相对位置(0~1)

        # ── 规则1: whitesmoke/blacksmoke 扁宽过滤 ──
        # 真实的烟(white/black)呈柱状/羽状，宽高比通常 < 1.5
        # 而建筑物/墙壁误检时通常是扁矩形 (aspect > 2.0)
        if cls_id in (0, 1, 3):   # blacksmoke, whitesmoke, smoke
            if aspect > 2.2 and conf < 0.6:
                return False
            if aspect > 3.0:
                return False

        # ── 规则2: 宽扁框靠近地面 → 建筑 ──
        # 宽扁框且底部靠近图像底部 → 很可能是建筑/墙壁
        if cls_id == 1:   # whitesmoke (最容易误检)
            if aspect > 1.8 and bottom_ratio > 0.75 and conf < 0.55:
                return False
            # 超宽框即使置信度高也过滤
            if aspect > 3.5:
                return False

        # ── 规则3: 面积过大/过小过滤 ──
        # 烟不可能占满整张图，也不可能小到几乎看不见
        if area_ratio > 0.85:       # 框占图 >85%
            return False
        if area_ratio < 0.002:      # 框占图 <0.2% (图片1000x1000时box<2000像素)
            if conf < 0.4:
                return False

        # ── 规则4: 位置过滤 ──
        # 烟/火焰不会出现在图片最底部边缘以下
        if cls_id in (0, 1, 2, 3):
            if y1 >= self.h - 5 and y2 >= self.h - 2:
                return False

        # ── 规则5: fire 尺寸约束 ──
        # 火焰不会太大（不占半张图），也不会是细长条
        if cls_id == 2:   # fire
            if aspect > 3.0:
                return False
            if area_ratio > 0.5:
                return False

        return True


# ─────────────────────────────────────────────
#  时序稳定模块  (防闪烁 / 去瞬时误检 / 提高报警可靠性)
# ─────────────────────────────────────────────

@dataclass
class Tracklet:
    """
    单个目标轨迹。

    核心设计:
      - 置信度迟滞: 出现阈值 > 消失阈值，防止阈值边界闪烁
      - 置信度衰减: 缺失帧时指数衰减，而非硬性计数
      - 预热期: 新轨迹需累积足够证据才输出，滤除瞬时误检
      - 可靠性评分: 综合 age、命中率、置信度稳定性，供报警使用
    """
    cls_id: int
    smoothed_box: list               # [x1, y1, x2, y2] EMA平滑后坐标
    smoothed_conf: float             # EMA平滑后置信度
    peak_conf: float                 # 历史最高置信度（记录证据强度）
    age: int = 0                     # 累计命中帧数（含预热期）
    consecutive_hits: int = 0        # 连续命中帧数（预热期计数器）
    missing: int = 0                 # 连续缺失帧数
    missing_conf_decay: float = 1.0  # 缺失帧时置信度衰减系数
    is_confirmed: bool = False       # 是否通过预热期
    hit_rate: float = 1.0            # 命中率 (age / total_frames_since_birth)
    birth_frame: int = 0             # 创建时的帧号
    alpha_box: float = 0.35          # 坐标EMA系数
    alpha_conf: float = 0.25         # 置信度EMA系数


class TemporalStabilizer:
    """
    时序稳定器 —— 去瞬时误检、防闪烁、提高报警可靠性。

    三大设计目标:
      1. 去瞬时误检: 新检测需经过预热期（连续命中 + 置信度积累）才输出
      2. 防闪烁:     置信度迟滞 + EMA平滑 + 逐轨迹闪烁检测
      3. 报警可靠性:  输出 reliability 评分（含时序稳定性因子）

    核心策略:
      - 置信度迟滞 (appear_thresh > disappear_thresh)
      - 预热期: 连续命中 min_hits 帧、且累积置信度达标后才输出
      - 缺失帧置信度指数衰减，衰减到阈值以下视为轨迹消亡
      - 逐轨迹闪烁检测: 记录 status 翻转次数，高频翻转的轨迹被抑制
    """

    def __init__(self,
                 iou_thresh: float = 0.30,
                 appear_thresh: float = 0.35,
                 disappear_thresh: float = 0.20,
                 min_hits: int = 3,
                 miss_ttl: int = 5,
                 flicker_suppress_thresh: int = 4,
                 flicker_window: int = 15):
        """
        Args:
            iou_thresh:       IoU匹配阈值
            appear_thresh:    置信度出现阈值 — 超过此值才可能输出（防止低置信度误检）
            disappear_thresh: 置信度消失阈值 — 低于此值才停止输出（防止闪烁消失）
            min_hits:         预热期帧数 — 连续命中这么多帧、且置信度达标才输出
            miss_ttl:         缺失保留帧数 — 轨迹丢失后保留的帧数
            flicker_suppress_thresh: 闪烁抑制阈值 — status 翻转超过此值则压制
            flicker_window:    闪烁检测窗口
        """
        self.iou_thresh = iou_thresh
        self.appear_thresh = appear_thresh
        self.disappear_thresh = disappear_thresh
        self.min_hits = min_hits
        self.miss_ttl = miss_ttl
        self.flicker_suppress_thresh = flicker_suppress_thresh
        self.flicker_window = flicker_window

        self.tracks: list[Tracklet] = []
        self.frame_count = 0
        # 逐轨迹闪烁检测器: track_id -> [status_history]
        self._track_flicker: dict[int, list[int]] = defaultdict(list)

    # ── 公开方法 ──────────────────────────

    def update(self, detections: list) -> list:
        """
        处理一帧检测结果。

        Args:
            detections: [(x1, y1, x2, y2, conf, cls_id), ...]

        Returns:
            [(x1, y1, x2, y2, conf, cls_id, reliability), ...]
            其中 reliability ∈ [0,1]，表示时序可靠性，可用于报警决策。
        """
        self.frame_count += 1
        used_track_idx = set()
        alive_tracks = []
        output = []

        # ── 1. 匹配: 当前检测 → 已有轨迹 ──
        matched_pairs = self._match_detections(detections)

        for det_idx, track_idx in matched_pairs:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            x1, y1, x2, y2, conf, cls_id = det
            used_track_idx.add(track_idx)

            # 更新框坐标 (EMA)
            track.smoothed_box = [
                track.alpha_box * n + (1 - track.alpha_box) * o
                for n, o in zip([x1, y1, x2, y2], track.smoothed_box)
            ]
            # 更新置信度 (EMA)
            track.smoothed_conf = (
                track.alpha_conf * conf + (1 - track.alpha_conf) * track.smoothed_conf
            )
            # 更新峰值置信度
            track.peak_conf = max(track.peak_conf, conf)
            # 更新计数
            track.consecutive_hits += 1
            track.age += 1
            track.missing = 0
            track.missing_conf_decay = 1.0
            # 预热期判定
            if (track.consecutive_hits >= self.min_hits
                    and track.smoothed_conf >= self.appear_thresh):
                track.is_confirmed = True

        # ── 2. 未匹配的旧轨迹: 缺失处理 ──
        for i, track in enumerate(self.tracks):
            if i not in used_track_idx:
                if track.age > 0:
                    track.missing += 1
                    track.consecutive_hits = 0  # 连续命中中断
                    # 置信度指数衰减 (每缺失一帧乘 0.7)
                    track.missing_conf_decay *= 0.7
                    track.smoothed_conf *= 0.7
                    # 若已确认的轨迹，在 miss_ttl 内保留；未确认的轨迹直接丢弃
                    if track.is_confirmed:
                        if track.missing <= self.miss_ttl:
                            alive_tracks.append(i)
                    else:
                        if track.missing <= 1:  # 未确认的只给1帧机会
                            alive_tracks.append(i)
                else:
                    alive_tracks.append(i)

        # ── 3. 未匹配的当前检测 → 创建新轨迹 ──
        used_det_idxs = {p[0] for p in matched_pairs}
        for i, det in enumerate(detections):
            if i in used_det_idxs:
                continue
            x1, y1, x2, y2, conf, cls_id = det
            # 新检测置信度必须达到出现阈值的一半才有资格创建轨迹
            # 避免极低置信度误检创建过多轨迹
            if conf < self.appear_thresh * 0.5:
                continue
            new_track = Tracklet(
                cls_id=cls_id,
                smoothed_box=[x1, y1, x2, y2],
                smoothed_conf=conf,
                peak_conf=conf,
                age=1,
                consecutive_hits=1,
                birth_frame=self.frame_count,
            )
            self.tracks.append(new_track)
            alive_tracks.append(len(self.tracks) - 1)

        # ── 4. 清理已确认但未标记为 alive 的轨迹 ──
        #    (这些是 matched 但未在步骤2标记的)
        for i, track in enumerate(self.tracks):
            if i in used_track_idx:
                alive_tracks.append(i)

        # ── 5. 逐轨迹闪烁检测 ──
        #    记录每个轨迹的 visible/hidden 翻转次数
        flicker_suppressed = set()
        now_visible = set(alive_tracks)
        for tid in now_visible:
            track = self.tracks[tid]
            hist = self._track_flicker[id(track)]
            hist.append(1)
            if len(hist) > self.flicker_window:
                hist.pop(0)
            # 统计翻转次数: 01 或 10 模式
            flips = sum(1 for j in range(1, len(hist)) if hist[j] != hist[j-1])
            if flips >= self.flicker_suppress_thresh:
                flicker_suppressed.add(tid)
        # 不在 visible 中的轨迹记录 0
        for track in self.tracks:
            if id(track) not in self._track_flicker:
                continue
            hist = self._track_flicker[id(track)]
            actual_visible = self.tracks.index(track) in now_visible
            if not actual_visible:
                hist.append(0)
                if len(hist) > self.flicker_window:
                    hist.pop(0)

        # ── 6. 构建输出 ──
        for i in alive_tracks:
            track = self.tracks[i]
            # 条件: 已确认 + 未闪烁抑制 + 平滑置信度 > 消失阈值
            if not track.is_confirmed:
                continue
            if i in flicker_suppressed:
                continue
            if track.smoothed_conf < self.disappear_thresh:
                continue

            bx1, by1, bx2, by2 = track.smoothed_box
            # 可靠性评分 (0~1, 越高越可靠)
            reliability = self._compute_reliability(track)

            output.append((
                bx1, by1, bx2, by2,
                track.smoothed_conf,
                track.cls_id,
                reliability,
            ))

        # ── 7. 最终清理: 移除过期轨迹 ──
        self._cleanup()

        return output

    def _match_detections(self, detections: list) -> list[tuple]:
        """
        贪心匹配: 当前检测 → 已有轨迹，基于 IoU + 类别。
        返回 [(det_idx, track_idx), ...]
        """
        matched = []
        used_tracks = set()

        # 按置信度降序处理检测（高置信度优先匹配）
        scored_dets = sorted(
            enumerate(detections),
            key=lambda x: x[1][4],  # conf
            reverse=True,
        )

        for det_idx, det in scored_dets:
            x1, y1, x2, y2, conf, cls_id = det
            best_iou = self.iou_thresh
            best_track = None

            for track_idx, track in enumerate(self.tracks):
                if track_idx in used_tracks:
                    continue
                if track.cls_id != cls_id:
                    continue
                if track.missing > self.miss_ttl:
                    continue

                tx1, ty1, tx2, ty2 = track.smoothed_box
                iou = self._compute_iou(x1, y1, x2, y2, tx1, ty1, tx2, ty2)
                # 对于已确认的轨迹，略微降低匹配门槛
                if track.is_confirmed:
                    iou *= 1.15
                if iou > best_iou:
                    best_iou = iou
                    best_track = track_idx

            if best_track is not None:
                matched.append((det_idx, best_track))
                used_tracks.add(best_track)

        return matched

    def _compute_reliability(self, track: Tracklet) -> float:
        """
        计算轨迹的时序可靠性评分 (0~1)，用于报警决策。

        因子:
          - age_factor:    累计命中帧数越多越可靠 (sigmoid-like)
          - hit_rate:      命中率越高越可靠
          - conf_stability: 峰值置信度与当前置信度的差距（差距小则稳定）
          - miss_penalty:  缺失帧数惩罚
        """
        # age 因子: 年龄越大越可靠，5帧后趋于饱和
        age_factor = 1.0 - 0.5 / (1.0 + track.age * 0.2)

        # 命中率因子
        total_frames = self.frame_count - track.birth_frame + 1
        hit_rate = track.age / max(total_frames, 1)
        hit_rate_factor = 0.3 + 0.7 * hit_rate

        # 置信度稳定性: 峰值与当前值接近 = 稳定
        if track.peak_conf > 0:
            conf_ratio = track.smoothed_conf / track.peak_conf
        else:
            conf_ratio = 1.0
        conf_stability = 0.5 + 0.5 * conf_ratio

        # 缺失惩罚
        miss_penalty = max(0.6, 1.0 - track.missing * 0.1)

        reliability = (
            age_factor * 0.35
            + hit_rate_factor * 0.25
            + conf_stability * 0.25
            + miss_penalty * 0.15
        )
        return max(0.0, min(1.0, reliability))

    def _compute_iou(self, ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        union = a_area + b_area - inter
        return inter / union if union > 0 else 0.0

    def _cleanup(self):
        """移除过期轨迹"""
        keep = []
        for track in self.tracks:
            if track.is_confirmed:
                # 已确认轨迹: 允许 miss_ttl 帧缺失
                if track.missing <= self.miss_ttl:
                    keep.append(track)
            else:
                # 未确认轨迹: 
                #   - 连续命中 < min_hits → 保留继续预热
                #   - 但总帧数超过 2*min_hits 仍未确认 → 抛弃（证据不足）
                if track.consecutive_hits < self.min_hits:
                    if track.age < self.min_hits * 3:
                        keep.append(track)
        self.tracks = keep

        # 清理闪烁记录中已不存在的轨迹
        valid_ids = {id(t) for t in self.tracks}
        self._track_flicker = {
            tid: hist for tid, hist in self._track_flicker.items()
            if tid in valid_ids
        }

    def reset(self):
        """重置状态 (切换视频/场景时调用)"""
        self.tracks.clear()
        self.frame_count = 0
        self._track_flicker.clear()
