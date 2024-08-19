# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import argparse
from typing import List, Any, Generator, Tuple, Optional, Dict
from dataclasses import dataclass, asdict

import numpy as np
import cv2
import yaml
from prettytable import PrettyTable


@dataclass
class ClassWiseMetric:
    name: str
    iou: float
    acc: float
    dice: float

    def as_metric_dict(self) -> Tuple[Dict, Dict, Dict]:
        return {self.name: self.iou}, {self.name: self.acc}, {self.name: self.dice}

    def as_line(self) -> List[str]:
        return [self.name, f"{self.iou:.4f}", f"{self.acc:.4f}", f"{self.dice:.4f}"]


@dataclass
class Metric:
    mean_acc: float
    mean_iou: float
    mean_dice: float
    iou: Dict[str, float]
    acc: Dict[str, float]
    dice: Dict[str, float]
    ata: Optional[float] = 0  # adjusted trunk accuracy

    _m = ['iou', 'acc', 'dice']

    def __str__(self):
        table = PrettyTable()
        table.field_names = ["Class", "IoU", "Accuracy", "Dice"]
        rows = [["mean", f"{self.mean_iou:.4f}", f"{self.mean_acc:.4f}", f"{self.mean_dice:.4f}"]] + [c.as_line() for c in self.get_cls_metrics()]
        table.add_rows(rows)
        if self.ata is not None:
            out = table.get_string() + "\n" + f"Adjusted trunk accuracy: {self.ata:.4f}"
        else:
            out = table.get_string()
        return out

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            metrics = yaml.safe_load(f)
        return cls(**metrics)

    def save(self, save_path: str) -> None:
        with open(save_path, 'w') as f:
            yaml.dump(asdict(self), f)

    def get_cls_metrics(self) -> List[ClassWiseMetric]:
        return [ClassWiseMetric(name, self.iou[name], self.acc[name], self.dice[name]) for name in self.iou]


@dataclass
class MetricWithMainObject(Metric):
    main_object_iou: float = 0
    main_object_acc: float = 0

    def __str__(self):
        out = super().__str__()
        out += f"\nMain object IoU: {self.main_object_iou:.4f}, acc: {self.main_object_acc:.4f}"
        return out


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def bbox_area(self) -> int:
        return self.w * self.h

    @property
    def bbox_center(self) -> np.ndarray:
        return np.array([self.x + self.w // 2, self.y + self.h // 2])

    def as_list(self, fmt: str = 'xywh') -> List[int]:
        if fmt not in ['xywh', 'xyxy']:
            raise ValueError(f"Invalid format: {fmt}! Accepted formats: xywh, xyxy")
        return [self.x, self.y, self.w, self.h] if fmt == 'xywh' else [self.x, self.y, self.x + self.w, self.y + self.h]

    def draw(self, image: np.ndarray) -> np.ndarray:
        cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h),
                      color=(255, 255, 0),
                      thickness=2)
        return image

    def slice_array(self, array: np.ndarray) -> np.ndarray:
        return array[self.y:self.y + self.h, self.x:self.x + self.w]

    def __str__(self) -> str:
        return f"(x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h})"


def expand_bbox(bbox: BBox,
                expand_ratio_x: float, expand_ratio_y: float,
                img_shape: Optional[Tuple[int, int]] = None) -> BBox:
    """
    Expand a bounding box.

    Args:
        bbox (BBox): The bounding box to expand.
        expand_ratio_x (float): The expand ratio for x-axis.
        expand_ratio_y (float): The expand ratio for y-axis.
        img_shape (Tuple[int, int]): The shape of the image. If None, the image shape is not considered.

    Returns:
        BBox: The expanded bounding box.
    """
    if expand_ratio_x < 0 or expand_ratio_y < 0:
        raise ValueError(f"expand_ratio_x and expand_ratio_y must be non-negative, but got {expand_ratio_x} and {expand_ratio_y}!")
    center = bbox.bbox_center
    new_w, new_h = int(bbox.w * (1 + expand_ratio_x)), int(bbox.h * (1 + expand_ratio_y))
    new_x, new_y = int(center[0] - new_w // 2), int(center[1] - new_h // 2)
    if img_shape is not None:
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(img_shape[1] - new_x, new_w)
        new_h = min(img_shape[0] - new_y, new_h)
    return BBox(new_x, new_y, new_w, new_h)


def get_largest_connected_component_in_center(bin_mask: np.ndarray, img_shape: Optional[Tuple[int, int]],
                                              ignore_x: float = 0.1, ignore_y: float = 0.1,
                                              ) -> Optional[Tuple[np.ndarray, BBox]]:
    """
    Get the largest connected component in the center area of the binary mask, ignoring the outside.

    Args:
        bin_mask (np.ndarray): The binary mask.
        img_shape (Tuple[int, int]): The shape of the image. If None, the image shape is not considered and thus do not consider the outside.
        ignore_x (float): The ratio of the width to ignore. Symmetrically ignore the left and right.
        ignore_y (float): The ratio of the height to ignore. Symmetrically ignore the top and bottom.

    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8))
    if num_labels == 1:
        return None
    if img_shape is None:
        area_max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    else:
        area_desc_order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
        for i in area_desc_order:
            if img_shape[1] * ignore_x < centroids[i][0] < img_shape[1] * (1 - ignore_x) and img_shape[0] * ignore_y < centroids[i][1] < img_shape[0] * (1 - ignore_y):
                area_max_label = i
                break
        else:
            return None
    target_bbox = BBox(*stats[area_max_label, :4])
    mask = (labels == area_max_label)
    return mask, target_bbox


def batch_spliter(lst: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """
    Split a list into batches.

    Args:
        lst (List[Any]): The list to split.
        batch_size (int): The batch size.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, but got {batch_size}!")
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def update_legacy_cfg(cfg):
    # obtained from DAFormer repo
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    if cfg.model.backbone.type == 'MobileNetV3':
        cfg.model.backbone.out_indices = tuple(cfg.model.backbone.out_indices)
    if cfg.model.decode_head.type == 'LRASPPHead':
        cfg.model.decode_head.in_channels = tuple(
            cfg.model.decode_head.in_channels)
        cfg.model.decode_head.in_index = tuple(cfg.model.decode_head.in_index)
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def add_parser_arguments(parser: argparse.ArgumentParser,
                         dataset: bool = True,
                         evaluation: bool = True,
                         output: bool = True,):
    if dataset:
        parser.add_argument_group("Dataset")
        parser.add_argument("--image_dir", type=str, default="./data/Apple_Farm_Real/images/validation")
        parser.add_argument("--img_extension", type=str, default="JPG", help="image extension for dataset")
        parser.add_argument("--anno_dir", type=str, default="./data/Apple_Farm_Real/annotations/validation")
        parser.add_argument("--depth_map_dir", type=str, default='/srv/dataset/Apple_Farm_Real_depth')
        parser.add_argument("--exclusion", type=str, nargs='+', default=["../daformer/tools/exclusion.txt"],
                            help="images to be excluded from evaluation")
        parser.add_argument("--exclude_air", action="store_true", help="exclude aerial images from evaluation")

    if evaluation:
        parser.add_argument_group("Evaluation")
        parser.add_argument("--ignore_class", type=str, nargs='+', default=["low-vegetation"],
                            help="classes to be ignored from evaluation")

        parser.add_argument("--with_main_object", action="store_true", help="evaluate with main object")
        parser.add_argument("--center_class", type=str, default="trunk", help="center class for the crop")
        parser.add_argument("--expand_ratio_x", type=float, default=0.3,
                            help="expand ratio for x-axis. 0.5 -> 50% expansion")
        parser.add_argument("--expand_ratio_y", type=float, default=0.2,
                            help="expand ratio for y-axis. 0.5 -> 50% expansion")

        parser.add_argument("--stat_groups", type=str, nargs='+',
                            default=["winter", "summer"],
                            help="grouping of statistics in accordance to different image categories (from image names)", )
        parser.add_argument("--depth_threshold",
                            type=float, nargs='+', default=[255],
                            help="depth thresholds for evaluation")

    if output:
        parser.add_argument_group("Output")
        parser.add_argument("--output",
                            nargs='+',
                            type=str,
                            choices=["npy", "conf", "img", "comp_img", "none"],
                            default=["none"],
                            help="output to be saved. img for semantic overlaid images, "
                                 "comp_img for prediction - annotation (- depth) comparison, overlaid.")
        parser.add_argument("--show_ori_img", action="store_true",
                            help="show original image when there is no depth map")
        parser.add_argument(
            '--out_dir', default='', help='directory where painted images will be saved')
        parser.add_argument(
            '--opacity',
            type=float,
            default=0.8,
            help='Opacity of painted segmentation map. In (0, 1] range.')
    

def extract_filestem(fstr: str) -> str:
    """
    Extracts a filename and return its stem (without extension)
    """
    return fstr[:fstr.rfind(".")]
