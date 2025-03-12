"""
DETR完整实现 v3.0
功能：端到端目标检测训练系统，支持DOTA数据集
模块：数据加载、模型构建、匈牙利损失、训练/测试流程
日期：2025-03-10
"""
import os
import json
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


# === 配置参数 ===
class Config:
    data_root = "./dota/train"
    train_ann = "./dota/train/annotations/train_coco.json"
    val_ann = "./dota/val/annotations/val_coco.json"
    num_classes = 15  # DOTA类别数
    hidden_dim = 256
    num_queries = 100
    nheads = 8
    enc_layers = 3
    dec_layers = 3
    lr = 1e-4
    batch_size = 4
    epochs = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "./outputs"


# === 数据加载 ===
class DOTA2COCO(torchvision.datasets.CocoDetection):
    def __init__(self, root, ann_file, transforms=None):
        if not os.path.exists(ann_file):
            self.convert_dota_to_coco(root, ann_file)  # 新增转换方法
        super().__init__(
            root=os.path.join(root, 'images'),
            annFile=ann_file,
            transforms=transforms
        )

    def convert_dota_to_coco(self, root, output_path):
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

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # === 图像预处理 ===
        def preprocess_image(image):
            # 转换PIL.Image到numpy数组
            img_array = np.array(image)

            # 动态缩放与填充
            h, w = img_array.shape[:2]
            scale = 1024 / max(h, w)
            resized_img = cv2.resize(img_array, (int(w * scale), int(h * scale)))

            # 右下填充
            pad_h = 1024 - resized_img.shape[0]
            pad_w = 1024 - resized_img.shape[1]
            padded_img = np.pad(resized_img, ((0, pad_h), (0, pad_w), (0, 0)),
                                mode='constant', constant_values=114)
            assert padded_img.shape == (1024, 1024, 3), "图像尺寸错误"
            # 转换为Tensor并归一化
            return torch.from_numpy(padded_img).permute(2, 0, 1).float() / 255.0

        # === 标注预处理 ===
        def preprocess_ann(ann, scale):
            # 调整框坐标
            x, y, w, h = ann['bbox']
            new_bbox = [
                x * scale,
                y * scale,
                w * scale,
                h * scale
            ]

            # 边界检查
            new_bbox[0] = min(new_bbox[0], 1024 - new_bbox[2])
            new_bbox[1] = min(new_bbox[1], 1024 - new_bbox[3])

            return {
                'boxes': torch.tensor([new_bbox], dtype=torch.float32),
                'labels': torch.tensor([ann['category_id']], dtype=torch.int64),
                'area': torch.tensor([ann['area'] * (scale ** 2)]),
                'iscrowd': torch.tensor([ann['iscrowd']])
            }

        # 执行预处理
        img_tensor = preprocess_image(img)
        scale = 1024 / max(img.height, img.width)

        # 整合标注
        processed_target = {}
        for key in ['boxes', 'labels', 'area', 'iscrowd']:
            processed_target[key] = torch.cat([
                preprocess_ann(ann, scale)[key] for ann in target
            ], dim=0) if len(target) > 0 else torch.empty((0,))

        processed_target['image_id'] = torch.tensor([target[0]['image_id']]) if target else torch.tensor([-1])

        return img_tensor, processed_target

    def __len__(self):
        return len(self.ids)


# === 模型架构 ===
class PositionEmbedding(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

    def forward(self, x):
        B, C, H, W = x.shape
        pos_x = torch.arange(W, device=x.device).reshape(1, 1, -1) * self.div_term
        pos_y = torch.arange(H, device=x.device).reshape(1, -1, 1) * self.div_term
        pos = torch.cat([pos_x.repeat(H, 1, 1), pos_y.repeat(1, W, 1)], dim=-1).permute(2, 0, 1)
        return pos.repeat(B, 1, 1, 1).reshape(B, self.d_model, H, W)


class DETR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Backbone
        # resnet = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True) 运行不出来
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.conv = nn.Conv2d(2048, cfg.hidden_dim, 1)

        # Transformer
        self.pos_encoder = PositionEmbedding(cfg.hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(cfg.hidden_dim, cfg.nheads, 2048),
            cfg.enc_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(cfg.hidden_dim, cfg.nheads, 2048),
            cfg.dec_layers
        )
        self.query_embed = nn.Embedding(cfg.num_queries, cfg.hidden_dim)

        # Heads
        self.class_head = nn.Linear(cfg.hidden_dim, cfg.num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)  # [B,2048,32,32]
        features = self.conv(features)  # [B,256,32,32]

        # Position encoding
        pos = self.pos_encoder(features) # [B,256,32,32]

        # Transformer
        src = features.flatten(2).permute(2, 0, 1) # [1024, B, 256]
        pos_embed = pos.flatten(2).permute(2, 0, 1)  # [1024, B, 256]

        # 维度校验
        assert src.shape == pos_embed.shape, \
            f"特征序列{src.shape}与位置编码{pos_embed.shape}不匹配"
        memory = self.encoder(src + pos_embed)

        query = self.query_embed.weight.unsqueeze(1).repeat(1, x.size(0), 1)
        hs = self.decoder(query, memory)

        # Predictions
        return {
            'pred_logits': self.class_head(hs),
            'pred_boxes': self.bbox_head(hs)
        }


# === 损失计算 ===
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_logits = outputs['pred_logits'].softmax(-1)  # [B,N,C+1]
        pred_boxes = outputs['pred_boxes']  # [B,N,4]
        batch_size = pred_logits.size(0)

        indices = []
        for b in range(batch_size):
            tgt_ids = targets[b]['labels']
            tgt_boxes = targets[b]['boxes']

            # Cost matrix
            cost_class = -pred_logits[b, :, tgt_ids]  # [N,M]
            cost_bbox = torch.cdist(pred_boxes[b], tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes[b]),
                box_cxcywh_to_xyxy(tgt_boxes)
            )
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

            # Hungarian matching
            row_idx, col_idx = linear_sum_assignment(C.cpu())
            indices.append((row_idx, col_idx))
        return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.alpha = 0.25  # Focal loss参数

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        src_logits = outputs['pred_logits']  # [B,N,C+1]
        src_boxes = outputs['pred_boxes']  # [B,N,4]

        # 分类损失
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.long, device=src_logits.device)
        for b, (row, col) in enumerate(indices):
            target_classes[b, row] = targets[b]['labels'][col]

        # Focal loss
        class_loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes,
                                     reduction='none', weight=torch.tensor([1.0] * self.num_classes + [0.1]))

        # 框回归损失
        idx = self._get_src_permutation_idx(indices)
        src_boxes_matched = src_boxes[idx]
        tgt_boxes_matched = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # L1 + GIoU
        l1_loss = F.l1_loss(src_boxes_matched, tgt_boxes_matched, reduction='none').sum(1)
        giou_loss = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes_matched),
            box_cxcywh_to_xyxy(tgt_boxes_matched)
        ))
        return (class_loss.mean() + l1_loss.mean() + giou_loss.mean()) / 3

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def det_collate_fn(batch):
    # 分离图像和目标
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # 堆叠图像张量 (假设所有图像已经预处理为相同尺寸)
    images = torch.stack(images, dim=0)

    # 为每个目标添加batch索引
    indexed_targets = []
    for batch_idx, target in enumerate(targets):
        if target['boxes'].numel() > 0:
            # 添加batch索引到boxes维度 [N,4] -> [N,5] (batch_idx, x1, y1, x2, y2)
            indexed_boxes = torch.cat([
                torch.full((target['boxes'].size(0), 1), batch_idx),  # 添加索引列
                target['boxes']
            ], dim=1)

            indexed_targets.append({
                'indexed_boxes': indexed_boxes,
                'labels': target['labels'],
                'image_id': target['image_id']
            })

    # 合并所有目标 (如果无目标则返回空张量)
    structured_targets = {
        'indexed_boxes': torch.cat([t['indexed_boxes'] for t in indexed_targets],
                                   dim=0) if indexed_targets else torch.empty((0, 5)),
        'labels': torch.cat([t['labels'] for t in indexed_targets], dim=0) if indexed_targets else torch.empty((0,)),
        'image_ids': torch.cat([t['image_id'] for t in indexed_targets], dim=0) if indexed_targets else torch.empty(
            (0,))
    }

    return images, structured_targets


# === 训练与测试 ===
def train(cfg):
    train_set = DOTA2COCO(cfg.data_root, cfg.train_ann)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, collate_fn=det_collate_fn, shuffle=True)

    model = DETR(cfg).to(cfg.device)
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

    test_set = DOTA2COCO(cfg.data_root, cfg.val_ann)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, collate_fn=det_collate_fn)

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
    # dota2coco = DOTA2COCO(cfg.data_root, cfg.train_ann)

    # 训练与评估
    train(cfg)
    # evaluate(cfg, os.path.join(cfg.output_dir, "detr_epoch49.pth"))
