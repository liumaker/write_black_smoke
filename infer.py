"""
黑烟白烟检测 - YOLO 推理脚本
支持单张图片、图片文件夹、视频、摄像头
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

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
    candidates = list(Path("runs/train").rglob("weights/best.pt"))
    if not candidates:
        print("[错误] 未找到模型文件 runs/train/**/weights/best.pt")
        print("请先运行 train.py 训练模型")
        sys.exit(1)
    return str(sorted(candidates)[-1])


def draw_boxes(img, results, show_label=True, show_conf=True):
    """在图片上绘制检测框"""
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            color = COLORS.get(cls_id, (0, 255, 0))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            if show_label or show_conf:
                text_parts = []
                if show_label:
                    text_parts.append(label)
                if show_conf:
                    text_parts.append(f"{conf:.2f}")
                text = " ".join(text_parts)

                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
    return img


def infer_image(model, image_path, output_dir, conf_thres, show):
    """推理单张图片"""
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[跳过] 文件不存在: {image_path}")
        return

    results = model(img_path, conf=conf_thres, iou=0.5)
    img = cv2.imread(str(img_path))
    img = draw_boxes(img, results)

    output_path = output_dir / img_path.name
    cv2.imwrite(str(output_path), img)
    print(f"  结果保存: {output_path}")

    if show:
        cv2.imshow("Inference", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def infer_video(model, video_path, output_dir, conf_thres, show):
    """推理视频文件"""
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

    frame_idx = 0
    print(f"视频信息: {w}x{h}, {fps}fps, {total}帧")
    print(f"推理中... (按 'q' 退出, 按 's' 暂停)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_thres, iou=0.5)
        frame = draw_boxes(frame, results)
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


def infer_webcam(model, conf_thres):
    """实时摄像头检测"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return

    print("摄像头实时检测中... (按 'q' 退出)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_thres, iou=0.5)
        frame = draw_boxes(frame, results)
        cv2.imshow("Inference - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="黑烟白烟检测 - YOLO 推理")
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
    parser.add_argument("--no-label", action="store_true",
                        help="不显示类别标签")
    parser.add_argument("--no-conf", action="store_true",
                        help="不显示置信度")
    args = parser.parse_args()

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
        print("  python infer.py video.mp4              # 视频文件")
        print("  python infer.py 0                      # 摄像头")
        print("  python infer.py --show image.jpg       # 显示结果")
        return

    print(f"\n推理源: {source}")
    print(f"置信度阈值: {args.conf}")
    print()

    if source == "0" or source == "camera":
        infer_webcam(model, args.conf)
    elif Path(source).is_dir():
        files = sorted(Path(source).glob("*"))
        for f in files:
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                infer_image(model, f, output_dir, args.conf, args.show)
    elif Path(source).suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        infer_video(model, source, output_dir, args.conf, args.show)
    else:
        infer_image(model, source, output_dir, args.conf, args.show)


if __name__ == "__main__":
    main()
