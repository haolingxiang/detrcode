from .detr import build
from .my_dqdetr import my_dqdetr


def build_model(args):
    return build(args)
