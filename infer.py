"""
黑烟白烟检测 - YOLO 推理/跟踪脚本
支持单张图片、图片文件夹、视频、摄像头、目标跟踪
集成规则过滤 + 时序稳定模块
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from smoke_filter import RuleFilter, TemporalStabilizer

# 类别配置 (与训练时保持一致)
CLASS_NAMES = {0: "blacksmoke", 1: "whitesmoke", 2: "fire", 3: "smoke"}
# 可视化颜色 (BGR)
COLORS = {
    0: (0, 0, 0),       # blacksmoke - 黑色
    1: (200, 200, 200), # whitesmoke - 浅灰
    2: (0, 0, 255),     # fire       - 红色
    3: (128, 128, 128), # smoke      - 灰色
}


def find_model():
    """在 YOLO 默认输出路径下寻找最佳模型"""
    search_paths = ["runs/train", "runs/detect/train"]
    candidates = []
    for sp in search_paths:
        candidates.extend(list(Path(sp).rglob("weights/best.pt")))
    if not candidates:
        print("[错误] 未找到模型文件")
        print("  搜索路径: runs/train/ 和 runs/detect/train/")
        print("请先运行 train.py 训练模型")
        sys.exit(1)
    return str(sorted(candidates)[-1])


def _extract_boxes(results) -> list:
    """从 ultralytics Results 中提取 (x1,y1,x2,y2,conf,cls_id)"""
    boxes_data = []
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            boxes_data.append((x1, y1, x2, y2, conf, cls_id))
    return boxes_data


def draw_filtered_boxes(img, boxes_data, show_label=True, show_conf=True,
                        show_reliability=False, show_track_id=False):
    """
    在图片上绘制滤波后的检测框。

    boxes_data 支持格式:
      - 6元素: (x1, y1, x2, y2, conf, cls_id)
      - 7元素: (x1, y1, x2, y2, conf, cls_id, reliability)
    """
    for item in boxes_data:
        if len(item) == 6:
            x1, y1, x2, y2, conf, cls_id = item
            reliability = None
        else:
            x1, y1, x2, y2, conf, cls_id, reliability = item

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        color = COLORS.get(cls_id, (0, 255, 0))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 第一行: 类别 + 置信度
        text_parts = []
        if show_label:
            text_parts.append(label)
        if show_conf:
            text_parts.append(f"{conf:.2f}")
        text_line1 = " ".join(text_parts)

        # 第二行: 可靠性 (供调试)
        text_line2 = None
        if show_reliability and reliability is not None:
            text_line2 = f"rel:{reliability:.2f}"

        # 绘制第一行
        if text_line1:
            (tw, th), _ = cv2.getTextSize(text_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, text_line1, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        # 绘制第二行 (显示在框下方)
        if text_line2:
            (tw2, th2), _ = cv2.getTextSize(text_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_bottom = y2 + th2 + 6
            cv2.rectangle(img, (x1, y2), (x1 + tw2 + 4, label_bottom), color, -1)
            cv2.putText(
                img, text_line2, (x1 + 2, y2 + th2 + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )
    return img


def infer_image(model, image_path, output_dir, conf_thres, show, enable_filter):
    """推理单张图片"""
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[跳过] 文件不存在: {image_path}")
        return

    results = model(img_path, conf=conf_thres, iou=0.5)
    img = cv2.imread(str(img_path))

    if enable_filter:
        boxes_data = _extract_boxes(results)
        h, w = img.shape[:2]
        rf = RuleFilter(h, w)
        boxes_data = rf.filter_boxes(boxes_data)
        img = draw_filtered_boxes(img, boxes_data)
    else:
        img = draw_filtered_boxes(img, _extract_boxes(results))

    output_path = output_dir / img_path.name
    cv2.imwrite(str(output_path), img)
    print(f"  结果保存: {output_path}")

    if show:
        cv2.imshow("Inference", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def infer_video(model, video_path, output_dir, conf_thres, show,
                track=False, tracker="botsort.yaml",
                enable_filter=True, enable_temporal=True,
                show_reliability=False):
    """推理或跟踪视频文件"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = output_dir / video_path.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # 时序稳定器 (跨帧)
    stabilizer = TemporalStabilizer() if enable_temporal else None

    mode_str = "跟踪" if track else "推理"
    frame_idx = 0
    print(f"视频信息: {w}x{h}, {fps}fps, {total}帧")
    print(f"{mode_str}中... (按 'q' 退出, 按 's' 暂停)")
    print(f"  规则过滤: {'开' if enable_filter else '关'} | 时序稳定: {'开' if enable_temporal else '关'}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 原始检测/跟踪
        if track:
            results = model.track(frame, conf=conf_thres, iou=0.5, tracker=tracker, persist=True)
        else:
            results = model(frame, conf=conf_thres, iou=0.5)

        # 提取boxes
        raw_boxes = _extract_boxes(results)

        # ── 规则过滤 ──
        if enable_filter:
            rf = RuleFilter(h, w)
            raw_boxes = rf.filter_boxes(raw_boxes)

        # ── 时序稳定 ──
        if stabilizer is not None:
            raw_boxes = stabilizer.update(raw_boxes)

        frame = draw_filtered_boxes(frame, raw_boxes, show_reliability=show_reliability)
        writer.write(frame)

        if show:
            cv2.imshow("Inference", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.waitKey(0)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  处理帧: {frame_idx}/{total}")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"结果保存: {output_path}")


def infer_webcam(model, conf_thres, show, track=False, tracker="botsort.yaml",
                 enable_filter=True, enable_temporal=True, show_reliability=False):
    """实时摄像头检测/跟踪"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    stabilizer = TemporalStabilizer() if enable_temporal else None
    mode_str = "跟踪" if track else "检测"

    print(f"摄像头实时{mode_str}中... (按 'q' 退出)")
    print(f"  规则过滤: {'开' if enable_filter else '关'} | 时序稳定: {'开' if enable_temporal else '关'}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        if track:
            results = model.track(frame, conf=conf_thres, iou=0.5, tracker=tracker, persist=True)
        else:
            results = model(frame, conf=conf_thres, iou=0.5)

        raw_boxes = _extract_boxes(results)

        if enable_filter:
            rf = RuleFilter(h, w)
            raw_boxes = rf.filter_boxes(raw_boxes)

        if stabilizer is not None:
            raw_boxes = stabilizer.update(raw_boxes)

        frame = draw_filtered_boxes(frame, raw_boxes, show_reliability=show_reliability)
        cv2.imshow("Inference - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="黑烟白烟检测 - YOLO 推理/跟踪")
    parser.add_argument("source", type=str, nargs="?", default=None,
                        help="推理源: 图片路径/文件夹路径/视频路径/0(摄像头)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="模型路径 (默认自动查找 best.pt)")
    parser.add_argument("--conf", "-c", type=float, default=0.25,
                        help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--output", "-o", type=str, default="infer_results",
                        help="输出目录 (默认: infer_results)")
    parser.add_argument("--show", "-s", action="store_true",
                        help="显示结果窗口")
    parser.add_argument("--track", "-t", action="store_true",
                        help="启用目标跟踪 (仅视频/摄像头)")
    parser.add_argument("--tracker", type=str, default="botsort.yaml",
                        help="跟踪器配置: botsort.yaml (默认) 或 bytetrack.yaml")
    parser.add_argument("--no-label", action="store_true",
                        help="不显示类别标签")
    parser.add_argument("--no-conf", action="store_true",
                        help="不显示置信度")
    parser.add_argument("--no-filter", action="store_true",
                        help="关闭规则过滤和时序稳定 (原始检测)")
    parser.add_argument("--show-reliability", action="store_true",
                        help="显示时序可靠性评分 (框下方)")
    args = parser.parse_args()

    enable_filter = not args.no_filter
    enable_temporal = not args.no_filter
    show_reliability = args.show_reliability

    model_path = args.model or find_model()
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = args.source
    if source is None:
        print("\n用法示例:")
        print("  python infer.py image.jpg              # 单张图片")
        print("  python infer.py images/                # 图片文件夹")
        print("  python infer.py video.mp4              # 推理视频")
        print("  python infer.py --track video.mp4      # 跟踪视频")
        print("  python infer.py 0                      # 摄像头检测")
        print("  python infer.py --track 0              # 摄像头跟踪")
        print("  python infer.py --show image.jpg       # 显示结果")
        print("  python infer.py --no-filter video.mp4  # 关闭后处理")
        return

    print(f"\n推理源: {source}")
    print(f"置信度阈值: {args.conf}")
    if args.track:
        print(f"跟踪模式: 启用 (tracker={args.tracker})")
    print()

    if source == "0" or source == "camera":
        infer_webcam(model, args.conf, args.show,
                     track=args.track, tracker=args.tracker,
                     enable_filter=enable_filter, enable_temporal=enable_temporal,
                     show_reliability=show_reliability)
    elif Path(source).is_dir():
        files = sorted(Path(source).glob("*"))
        for f in files:
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                infer_image(model, f, output_dir, args.conf, args.show,
                            enable_filter=enable_filter)
    elif Path(source).suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        infer_video(model, source, output_dir, args.conf, args.show,
                    track=args.track, tracker=args.tracker,
                    enable_filter=enable_filter, enable_temporal=enable_temporal,
                    show_reliability=show_reliability)
    else:
        infer_image(model, source, output_dir, args.conf, args.show,
                    enable_filter=enable_filter)


if __name__ == "__main__":
    main()
