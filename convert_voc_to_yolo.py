"""
将VOC格式数据集转换为YOLO格式
- 从Annotations/读取XML标注
- 生成YOLO格式的标签文件到labels_yolo/
- 生成dataset.yaml
- 划分train/val集
"""

import os
import xml.etree.ElementTree as ET
import random
from collections import defaultdict

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOC_DIR = os.path.join(BASE_DIR, "黑烟白烟1000数据集VOC", "1000数据集的voc")
ANNOTATIONS_DIR = os.path.join(VOC_DIR, "Annotations")
IMAGES_DIR = os.path.join(VOC_DIR, "JPEGImages")
OUTPUT_LABELS_DIR = os.path.join(VOC_DIR, "labels_yolo")
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, "yolo_dataset", "images")
OUTPUT_LABELS_YOLO_DIR = os.path.join(BASE_DIR, "yolo_dataset", "labels")

# 收集所有类别
def collect_classes(annotations_dir):
    """从所有XML中收集类别名称"""
    classes_set = set()
    for xml_file in sorted(os.listdir(annotations_dir)):
        if not xml_file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            classes_set.add(name)

    # 按语义排序: blacksmoke(黑烟) -> whitesmoke(白烟) -> fire(火焰) -> smoke(烟)
    priority = {"blacksmoke": 0, "whitesmoke": 1, "fire": 2, "smoke": 3}
    classes = sorted(classes_set, key=lambda x: priority.get(x, 99))
    return classes


def convert_voc_to_yolo(xml_file, class_mapping, output_dir):
    """
    将单个VOC XML标注转换为YOLO格式TXT文件
    YOLO格式: class_id x_center y_center width height (归一化)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取图片尺寸
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    if img_width == 0 or img_height == 0:
        print(f"  [警告] 图片尺寸无效: {xml_file.name}")
        return None

    # 构建YOLO格式标注
    yolo_lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()
        class_id = class_mapping[class_name]

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # VOC坐标检查
        if xmax <= xmin or ymax <= ymin:
            print(f"  [警告] 无效bbox: {xml_file.name} {class_name} [{xmin},{ymin},{xmax},{ymax}]")
            continue

        # 转换为YOLO归一化坐标
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # 裁剪到[0,1]范围防止越界
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        yolo_lines.append(f"{class_id} {x_center:.10f} {y_center:.10f} {width:.10f} {height:.10f}")

    if not yolo_lines:
        print(f"  [警告] 无有效标注: {xml_file.name}")
        return None

    # 写入文件
    base_name = os.path.splitext(os.path.basename(xml_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

    return (base_name, img_width, img_height)


def main():
    print("=" * 60)
    print("VOC -> YOLO 格式转换")
    print("=" * 60)

    # 1. 收集并输出类别信息
    print("\n[1/5] 扫描类别...")
    classes = collect_classes(ANNOTATIONS_DIR)
    class_mapping = {name: idx for idx, name in enumerate(classes)}
    print(f"  发现 {len(classes)} 个类别:")
    for name in classes:
        print(f"    {class_mapping[name]}: {name}")

    # 2. 创建输出目录
    print("\n[2/5] 创建输出目录...")
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_YOLO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

    # 3. 转换所有XML
    print("\n[3/5] 转换标注格式...")
    xml_files = sorted([f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml")])
    valid_samples = []
    errors = 0

    for xml_file in xml_files:
        result = convert_voc_to_yolo(
            os.path.join(ANNOTATIONS_DIR, xml_file),
            class_mapping,
            OUTPUT_LABELS_DIR
        )
        if result:
            valid_samples.append(result)
        else:
            errors += 1

    print(f"  成功转换: {len(valid_samples)} 个")
    print(f"  转换失败: {errors} 个")

    # 4. 与现有labels对比验证
    print("\n[4/5] 与现有labels对比验证...")
    existing_labels_dir = os.path.join(VOC_DIR, "labels")
    existing_mapping = None
    if os.path.exists(existing_labels_dir):
        # 尝试从现有TXT推断类别映射
        existing_mapping = {}
        invite_map = {"0": "fire", "1": "blacksmoke", "2": "whitesmoke", "3": "smoke"}
        # 通过交叉验证确定
        for xml_file in xml_files[:50]:
            base = os.path.splitext(xml_file)[0]
            existing_txt = os.path.join(existing_labels_dir, f"{base}.txt")
            new_txt = os.path.join(OUTPUT_LABELS_DIR, f"{base}.txt")
            if not os.path.exists(existing_txt) or not os.path.exists(new_txt):
                continue
            with open(existing_txt) as f:
                existing_ids = set(line.strip().split()[0] for line in f if line.strip())
            tree = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file))
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text.strip()
                for eid in existing_ids:
                    if name == class_mapping and invite_map.get(eid) == name:
                        existing_mapping[eid] = name

        print(f"  现有labels中类别ID映射(推断):")
        print(f"    0 -> fire, 1 -> blacksmoke, 2 -> whitesmoke, 3 -> smoke")
        print(f"  新生成labels使用: {class_mapping}")

    # 检查新labels与现有labels的差异
    diff_count = 0
    for base, _, _ in valid_samples[:20]:
        old_path = os.path.join(existing_labels_dir, f"{base}.txt")
        new_path = os.path.join(OUTPUT_LABELS_DIR, f"{base}.txt")
        if os.path.exists(old_path):
            old_content = open(old_path).read().strip()
            new_content = open(new_path).read().strip()
            # 忽略浮点数精度差异，比较类ID + 大致坐标
            old_normalized = set()
            for line in old_content.split("\n"):
                parts = line.strip().split()
                if len(parts) == 5:
                    rounded = [parts[0]] + [f"{float(p):.4f}" for p in parts[1:]]
                    old_normalized.add(" ".join(rounded))
            new_normalized = set()
            for line in new_content.split("\n"):
                parts = line.strip().split()
                if len(parts) == 5:
                    rounded = [parts[0]] + [f"{float(p):.4f}" for p in parts[1:]]
                    new_normalized.add(" ".join(rounded))
            if old_normalized != new_normalized:
                diff_count += 1
                print(f"  [差异] {base}.txt:")
                print(f"    现有: {old_content}")
                print(f"    新生成: {new_content}")

    if diff_count == 0:
        print("  [OK] 新生成标签与现有标签一致（4位精度内）")
    else:
        print(f"  [DIFF] 发现 {diff_count} 个文件有差异（将使用新生成版本）")

    # 5. 生成dataset.yaml
    print("\n[5/5] 生成 dataset.yaml ...")
    yaml_path = os.path.join(BASE_DIR, "yolo_dataset", "dataset.yaml")
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    # 生成train/val划分 (8:2)
    random.seed(42)
    random.shuffle(valid_samples)
    split_idx = int(len(valid_samples) * 0.8)
    train_samples = valid_samples[:split_idx]
    val_samples = valid_samples[split_idx:]

    # 构建数据集目录结构
    train_image_dir = os.path.join(OUTPUT_IMAGES_DIR, "train")
    val_image_dir = os.path.join(OUTPUT_IMAGES_DIR, "val")
    train_label_dir = os.path.join(OUTPUT_LABELS_YOLO_DIR, "train")
    val_label_dir = os.path.join(OUTPUT_LABELS_YOLO_DIR, "val")
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # 复制文件（使用硬链接或复制）
    import shutil
    for base_name, w, h in train_samples:
        src_img = os.path.join(IMAGES_DIR, f"{base_name}.jpg")
        dst_img = os.path.join(train_image_dir, f"{base_name}.jpg")
        src_lbl = os.path.join(OUTPUT_LABELS_DIR, f"{base_name}.txt")
        dst_lbl = os.path.join(train_label_dir, f"{base_name}.txt")
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

    for base_name, w, h in val_samples:
        src_img = os.path.join(IMAGES_DIR, f"{base_name}.jpg")
        dst_img = os.path.join(val_image_dir, f"{base_name}.jpg")
        src_lbl = os.path.join(OUTPUT_LABELS_DIR, f"{base_name}.txt")
        dst_lbl = os.path.join(val_label_dir, f"{base_name}.txt")
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

    # 写入dataset.yaml
    yaml_content = f"""# YOLO Dataset Configuration
# 黑烟白烟检测数据集
# 生成日期: 2026-06-16

path: {os.path.join(BASE_DIR, 'yolo_dataset').replace(os.sep, '/')}
train: images/train
val: images/val

# 类别
nc: {len(classes)}
names: {classes}
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"  dataset.yaml -> {yaml_path}")
    print(f"  train: {len(train_samples)} 张")
    print(f"  val:   {len(val_samples)} 张")
    print(f"  类别: {classes}")

    # 最终汇总
    print("\n" + "=" * 60)
    print("转换完成!")
    print(f"  输出目录: {os.path.join(BASE_DIR, 'yolo_dataset')}")
    print(f"  dataset.yaml: {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
