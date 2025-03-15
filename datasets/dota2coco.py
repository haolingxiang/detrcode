import json
import os

import cv2
from torchvision.datasets import CocoDetection
from tqdm import tqdm

from datasets.coco import ConvertCocoPolysToMask


class DOTA2COCODataset(CocoDetection):
    def __init__(self, root, ann_file, transforms, return_masks):
        if not os.path.exists(ann_file):
            convert_dota_to_coco(root, ann_file)  # 转换coco标注方法
        super(DOTA2COCODataset, self).__init__(os.path.join(root, 'images'), ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(DOTA2COCODataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)


# === 转换dota数据为coco形式 ===
def convert_dota_to_coco(root, output_path):
    """
    实现DOTA原生格式到COCO格式的转换
    输入结构：
    root/
      ├── images/  # 存放所有图像文件
      └── labelTxt/  # 存放DOTA格式的txt标注文件
    """
    # 初始化COCO结构
    coco_dict = {
        "info": {"description": "DOTA-COCO Format"},
        "images": [],
        "annotations": [],
        "categories": []
    }

    # === 1. 构建类别映射 ===
    class_names = [
        'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
        'basketball-court', 'ground-track-field', 'harbor', 'bridge',
        'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout',
        'soccer-ball-field', 'swimming-pool'
    ]
    coco_dict["categories"] = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]

    # === 2. 遍历所有图像 ===
    img_dir = os.path.join(root, "images")
    ann_dir = os.path.join(root, "labelTxt")
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]

    ann_id = 0
    for img_id, img_file in enumerate(tqdm(img_files)):
        # 获取图像尺寸
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # 添加图像记录
        coco_dict["images"].append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        # === 3. 解析DOTA标注 ===
        ann_file = os.path.join(ann_dir, img_file.replace('.png', '.txt').replace('.jpg', '.txt'))
        if not os.path.exists(ann_file):
            continue

        with open(ann_file, 'r') as f:
            lines = [line.strip().split() for line in f if line.startswith('gsd') is False]

        for line in lines:
            if len(line) < 9: continue

            # 解析DOTA多边形坐标（8点表示法）
            points = list(map(float, line[:8]))
            category = line[8]
            difficult = int(line[9]) if len(line) > 9 else 0

            # === 4. 转换为COCO格式 ===
            # 计算最小外接矩形（HBB）
            x_coords = points[::2]
            y_coords = points[1::2]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            w = x_max - x_min
            h = y_max - y_min

            # 添加标注记录
            coco_dict["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_names.index(category) + 1,
                "bbox": [x_min, y_min, w, h],
                "area": w * h,
                "iscrowd": difficult,
                "segmentation": [points]  # 保留原始多边形信息[1](@ref)
            })
            ann_id += 1

    # === 5. 保存文件 ===
    try:
        with open(output_path, 'w') as f:
            json.dump(coco_dict, f)
    except Exception as e:
        print(f"创建失败：{str(e)}")