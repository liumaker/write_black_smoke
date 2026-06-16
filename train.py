"""
黑烟白烟检测 - YOLO 训练脚本
基于 Ultralytics YOLOv8/v11
"""

import os
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """检查并安装依赖"""
    required = ["ultralytics", "torch", "torchvision"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[安装依赖] 缺少: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade"] + missing
        )
        print("[安装依赖] 完成")
    else:
        print("[检查依赖] 所有依赖已就绪")


def train():
    from ultralytics import YOLO

    DATA_YAML = "yolo_dataset/dataset.yaml"

    print("=" * 60)
    print("黑烟白烟检测 - YOLO 训练")
    print("=" * 60)
    print(f"数据集: {DATA_YAML}")
    print()

    # 加载预训练模型
    model = YOLO("yolov8m.pt")

    # 开始训练 (不指定 project/name，使用 YOLO 默认路径 runs/train/exp...)
    results = model.train(
        data=DATA_YAML,
        epochs=200,
        patience=30,
        batch=16,
        imgsz=640,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=10,
        device="0" if __import__("torch").cuda.is_available() else "cpu",
        workers=4,
        seed=42,
        val=True,
        save=True,
        save_period=10,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    return results


if __name__ == "__main__":
    check_dependencies()
    train()
