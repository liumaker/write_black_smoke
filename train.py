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

    # 项目根目录
    BASE_DIR = Path(__file__).parent.resolve()

    # 数据集配置路径
    DATA_YAML = str(BASE_DIR / "yolo_dataset" / "dataset.yaml")

    # 输出目录
    PROJECT = str(BASE_DIR / "runs")
    NAME = "smoke_detection"

    print("=" * 60)
    print("黑烟白烟检测 - YOLO 训练")
    print("=" * 60)
    print(f"数据集: {DATA_YAML}")
    print(f"输出目录: {PROJECT}/{NAME}")
    print()

    # 加载预训练模型 (推荐 yolov8m.pt 平衡速度与精度)
    # 可选: yolov8n.pt (轻量), yolov8s.pt, yolov8m.pt (推荐), yolov8l.pt, yolov8x.pt
    model = YOLO("yolov8m.pt")

    # 开始训练
    results = model.train(
        data=DATA_YAML,
        epochs=200,                # 训练轮数
        patience=30,               # early stopping 耐心值
        batch=16,                  # 根据显存调整
        imgsz=640,                 # 输入图片尺寸
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",          # 自动选择优化器
        lr0=0.01,                  # 初始学习率
        lrf=0.01,                  # 最终学习率 (余弦退火)
        momentum=0.937,            # SGD动量
        weight_decay=0.0005,       # 权重衰减
        warmup_epochs=3.0,         # 预热轮数
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,                   # box loss 增益
        cls=0.5,                   # cls loss 增益
        dfl=1.5,                   # dfl loss 增益
        hsv_h=0.015,               # 颜色增强
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,               # 不旋转 (烟通常垂直)
        translate=0.1,             # 平移
        scale=0.5,                 # 缩放
        shear=0.0,                 # 不剪切
        perspective=0.0,
        flipud=0.0,                # 不上下翻转 (烟的方向有意义)
        fliplr=0.5,                # 左右翻转
        mosaic=1.0,                # mosaic增强
        mixup=0.1,                 # mixup增强
        copy_paste=0.1,            # copy-paste增强
        close_mosaic=10,           # 最后10轮关闭mosaic
        device="0" if os.system("nvidia-smi > nul 2>&1") == 0 else "cpu",
        workers=4,
        seed=42,
        deterministic=False,
        val=True,
        save=True,
        save_period=10,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print(f"训练完成! 结果保存在: {PROJECT}/{NAME}")
    print("=" * 60)

    # 输出最佳模型路径
    best_pt = Path(PROJECT) / NAME / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n最佳模型: {best_pt}")

    return results


if __name__ == "__main__":
    check_dependencies()
    train()
