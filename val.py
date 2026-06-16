"""
黑烟白烟检测 - YOLO 验证脚本
对验证集进行评估，输出 mAP、召回率、精确率等指标
"""

from pathlib import Path
from ultralytics import YOLO


def find_best_model():
    """在 YOLO 默认输出路径下寻找最佳模型"""
    candidates = list(Path("runs/train").rglob("weights/best.pt"))
    if not candidates:
        print("[错误] 未找到训练好的模型 (runs/train/**/weights/best.pt)")
        print("请先运行 train.py 训练模型")
        return None
    # 取最新的模型
    return str(sorted(candidates)[-1])


def validate():
    DATA_YAML = "yolo_dataset/dataset.yaml"

    model_path = find_best_model()
    if model_path is None:
        return None

    print(f"加载模型: {model_path}")
    print(f"数据集: {DATA_YAML}")
    print()

    model = YOLO(model_path)

    # 不指定 project，使用 YOLO 默认路径 runs/val/exp...
    results = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=640,
        batch=16,
        device="0" if __import__("torch").cuda.is_available() else "cpu",
        conf=0.001,
        iou=0.6,
        max_det=300,
        save_json=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("验证结果摘要")
    print("=" * 60)
    print(f"  mAP@0.5:     {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"  精确率:        {results.box.mp:.4f}")
    print(f"  召回率:        {results.box.mr:.4f}")

    if hasattr(results.box, "ap_class_index") and hasattr(results.box, "maps"):
        class_names = ['blacksmoke', 'whitesmoke', 'fire', 'smoke']
        print("\n  各类别 AP@0.5:")
        for cls_id, ap in zip(results.box.ap_class_index, results.box.maps):
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            print(f"    {cls_name}: {ap:.4f}")

    print("=" * 60)
    return results


if __name__ == "__main__":
    validate()
