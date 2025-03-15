from pathlib import Path

from .coco import make_coco_transforms
from .dota2coco import DOTA2COCODataset


def build(image_set, args):
    root = Path(args.dota_path)
    assert root.exists(), f'provided DOTA path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / "annotations" / f'{mode}_train.json'),
        "val": (root / "val", root / "annotations" / f'{mode}_val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = DOTA2COCODataset(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
