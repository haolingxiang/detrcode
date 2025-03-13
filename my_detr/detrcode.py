"""
DETR完整实现 v3.0
功能：端到端目标检测训练系统，支持DOTA数据集
模块：数据加载、模型构建、匈牙利损失、训练/测试流程
日期：2025-03-10
"""
import os
import json
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm
import torchvision.models as models
from torchvision.datasets import CocoDetection
import datasets.transforms as T
from models.detr import DETR, SetCriterion
from models.matcher import HungarianMatcher
from util.misc import collate_fn, nested_tensor_from_tensor_list, NestedTensor


# === 配置参数 ===
class Config:
    data_root = "./dota/train"
    train_ann = "./dota/train/annotations/train_coco.json"
    val_ann = "./dota/val/annotations/val_coco.json"
    backbone = 'resnet50'
    num_classes = 15  # DOTA类别数
    hidden_dim = 256
    num_queries = 100
    nheads = 8
    enc_layers = 3
    dec_layers = 3
    dim_feedforward = 2048
    lr = 1e-4
    lr_backbone = 1e-5
    batch_size = 2
    epochs = 50
    dropout = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "./outputs"


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


class DOTA2COCODataset(CocoDetection):
    def __init__(self, root, ann_file, transforms=None):
        if not os.path.exists(ann_file):
            convert_dota_to_coco(root, ann_file)  # 转换coco标注方法
        super(DOTA2COCODataset, self).__init__(os.path.join(root, 'images'), ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask()

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


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')



# === 训练与测试 ===
def train(cfg):
    train_set = DOTA2COCODataset(cfg.data_root, cfg.train_ann, transforms=make_coco_transforms("train"))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, collate_fn=collate_fn, shuffle=True)

    model = DETR(cfg.backbone, cfg.transformer, ).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = SetCriterion(cfg.num_classes, HungarianMatcher())

    for epoch in range(cfg.epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(cfg.device)
            outputs = model(images)

            # 数据格式适配
            targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 保存模型
        torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"detr_epoch{epoch}.pth"))


def evaluate(cfg, model_path):
    model = DETR(cfg).to(cfg.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_set = DOTA2COCODataset(cfg.data_root, cfg.val_ann)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, collate_fn=collate_fn)

    coco_gt = COCO(os.path.join(cfg.data_root, cfg.val_ann))
    coco_dt = []

    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images.to(cfg.device))
            # 转换预测结果
            for b in range(images.size(0)):
                boxes = outputs['pred_boxes'][b].cpu()
                scores = outputs['pred_logits'][b].softmax(-1)[:, :-1].max(1)[0]
                labels = outputs['pred_logits'][b].argmax(1).cpu()

                for box, score, label in zip(boxes, scores, labels):
                    if score < 0.5: continue  # 过滤低置信度
                    coco_dt.append({
                        "image_id": targets[b]['image_id'].item(),
                        "category_id": label.item(),
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "score": score.item()
                    })

    # COCO评估
    with open("temp.json", 'w') as f:
        json.dump(coco_dt, f)
    coco_dt = coco_gt.loadRes("temp.json")
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


# === 辅助函数 ===
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


def generalized_box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union

    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]

    return iou - (enclose_area - union) / enclose_area


if __name__ == "__main__":
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 转换数据格式为coco形式
    # dota2coco = DOTA2COCODataset(cfg.data_root, cfg.train_ann)

    # 训练与评估
    train(cfg)
    # evaluate(cfg, os.path.join(cfg.output_dir, "detr_epoch49.pth"))
