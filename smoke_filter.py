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
#  时序稳定模块  (抗抖动 / 闪烁)
# ─────────────────────────────────────────────

@dataclass
class Tracklet:
    """单个目标轨迹"""
    cls_id: int
    smoothed_box: list           # [x1, y1, x2, y2] 平滑后
    smoothed_conf: float
    age: int = 0                 # 连续看到的帧数
    missing: int = 0             # 连续缺失帧数
    alpha_box: float = 0.4       # 坐标EMA系数
    alpha_conf: float = 0.3      # 置信度EMA系数


class TemporalStabilizer:
    """
    时序稳定器。
    解决: smoke框乱跳、fire/smoke检测不稳定。

    策略:
      - EMA平滑框坐标和置信度
      - 短生命周期检测抑制 (< min_hits 帧不显示)
      - 缺失帧恢复 (最多容忍 miss_ttl 帧)
      - 闪烁检测: 高频交替出现/消失时直接压制
    """

    def __init__(self, iou_thresh: float = 0.35,
                 min_hits: int = 2,
                 miss_ttl: int = 3,
                 flicker_window: int = 8):
        """
        iou_thresh:    IoU匹配阈值(当前检测与历史轨迹)
        min_hits:      最少连续命中帧数才显示
        miss_ttl:      轨迹缺失后保留帧数
        flicker_window:闪烁检测窗口
        """
        self.iou_thresh = iou_thresh
        self.min_hits = min_hits
        self.miss_ttl = miss_ttl
        self.flicker_window = flicker_window

        self.tracks: list[Tracklet] = []
        self.frame_count = 0
        self._flicker_counter = defaultdict(int)   # (cls_id,) -> 突变计数

    def update(self, detections: list) -> list:
        """
        输入: [(x1, y1, x2, y2, conf, cls_id), ...]
        输出: 稳定后的列表
        """
        self.frame_count += 1
        used_track_idx = set()
        new_detections = []

        # ── 1. 将当前检测与已有轨迹匹配 ──
        unmatched_det = list(detections)
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            best_idx, best_iou = self._find_best_match(x1, y1, x2, y2, cls_id)

            if best_idx is not None and best_iou >= self.iou_thresh:
                track = self.tracks[best_idx]
                # EMA平滑框坐标
                new_box = [x1, y1, x2, y2]
                track.smoothed_box = [
                    track.alpha_box * n + (1 - track.alpha_box) * o
                    for n, o in zip(new_box, track.smoothed_box)
                ]
                # EMA平滑置信度
                track.smoothed_conf = (
                    track.alpha_conf * conf + (1 - track.alpha_conf) * track.smoothed_conf
                )
                track.age += 1
                track.missing = 0
                used_track_idx.add(best_idx)
                unmatched_det.remove(det)
            else:
                # ── 2. 未匹配的检测 → 创建新轨迹 ──
                # 先检查这个检测本身是否已经通过规则过滤
                new_track = Tracklet(
                    cls_id=cls_id,
                    smoothed_box=[x1, y1, x2, y2],
                    smoothed_conf=conf,
                    age=1,
                )
                self.tracks.append(new_track)

        # ── 3. 未匹配的旧轨迹: 增加missing计数 ──
        for i, track in enumerate(self.tracks):
            if i not in used_track_idx and track.age > 0:
                track.missing += 1

        # ── 4. 清理过期轨迹 ──
        self.tracks = [t for t in self.tracks
                       if t.missing <= self.miss_ttl or t.age < self.min_hits]

        # ── 5. 闪烁检测 ──
        # 记录每个类别的活跃轨迹数变化，若高频抖动则压制
        active_by_class = defaultdict(int)
        for t in self.tracks:
            if t.age >= self.min_hits and t.missing == 0:
                active_by_class[t.cls_id] += 1

        flicker_suppress = set()
        for cls_id in (0, 1, 2, 3):
            key = (cls_id,)
            prev = self._flicker_counter[key]
            curr = active_by_class.get(cls_id, 0)
            if abs(curr - prev) >= 2:   # 活跃数突变
                self._flicker_counter[key] += 1
            else:
                self._flicker_counter[key] = max(0, self._flicker_counter[key] - 1)

            if self._flicker_counter[key] > self.flicker_window // 2:
                flicker_suppress.add(cls_id)

        # ── 6. 输出稳定的检测结果 ──
        for track in self.tracks:
            if track.age >= self.min_hits and track.missing == 0:
                if track.cls_id in flicker_suppress:
                    continue
                bx1, by1, bx2, by2 = track.smoothed_box
                new_detections.append(
                    (bx1, by1, bx2, by2, track.smoothed_conf, track.cls_id)
                )

        return new_detections

    def _find_best_match(self, x1, y1, x2, y2, cls_id):
        """根据IoU寻找最佳匹配轨迹"""
        best_iou = 0.0
        best_idx = None
        det_area = (x2 - x1) * (y2 - y1)
        for i, track in enumerate(self.tracks):
            if track.cls_id != cls_id:
                continue
            if track.missing > self.miss_ttl:
                continue
            tx1, ty1, tx2, ty2 = track.smoothed_box
            # IoU计算
            ix1, iy1 = max(x1, tx1), max(y1, ty1)
            ix2, iy2 = min(x2, tx2), min(y2, ty2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            track_area = (tx2 - tx1) * (ty2 - ty1)
            union = det_area + track_area - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        return best_idx, best_iou

    def reset(self):
        """重置状态 (切换视频/场景时调用)"""
        self.tracks.clear()
        self.frame_count = 0
        self._flicker_counter.clear()
