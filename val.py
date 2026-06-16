"""
黑烟白烟检测 - YOLO 验证脚本
对验证集进行评估，输出 mAP、召回率、精确率等指标
"""

from pathlib import Path
from ultralytics import YOLO


def validate():
    BASE_DIR = Path(__file__).parent.resolve()
    DATA_YAML = str(BASE_DIR / "yolo_dataset" / "dataset.yaml")

    # 自动寻找最佳模型
    candidates = list(BASE_DIR.glob("runs/smoke_detection/weights/best.pt"))
    if not candidates:
        # 尝试在同类目录下查找最近训练的模型
        candidates = list(BASE_DIR.glob("runs/**/weights/best.pt"))

    if not candidates:
        print("[错误] 未找到训练好的模型 (runs/**/weights/best.pt)")
        print("请先运行 train.py 训练模型")
        return None

    model_path = str(candidates[0])
    print(f"加载模型: {model_path}")
    print(f"数据集: {DATA_YAML}")
    print()

    model = YOLO(model_path)

    results = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=640,
        batch=16,
        device="0" if __import__("torch").cuda.is_available() else "cpu",
        project=str(BASE_DIR / "runs"),
        name="smoke_validation",
        exist_ok=True,
        conf=0.001,
        iou=0.6,
        max_det=300,
        save_json=True,
        save_hybrid=False,
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
        for i, (cls_id, ap) in enumerate(zip(results.box.ap_class_index, results.box.maps)):
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            print(f"    {cls_name}: {ap:.4f}")

    print(f"\n  结果保存至: runs/smoke_validation/")
    print("=" * 60)

    return results


if __name__ == "__main__":
    validate()
