import pathlib
from typing import Dict, Union, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import torch
import numpy as np
import cv2

from .types import Classes, Palette
from .crop import Crop
from .utils import expand_bbox, get_largest_connected_component_in_center, Metric, MetricWithMainObject, ClassWiseMetric
from .visualize import save_results, initialize_figures, get_piecewise_cmap


@dataclass
class ClassWiseStatistics(ClassWiseMetric):
    _counter: int = 0

    @property
    def counter(self) -> int:
        return self._counter

    def count(self) -> None:
        self._counter += 1


def zero_dict():
    return defaultdict(int)


@dataclass
class Statistics:
    classes: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.mean_iou: float = 0.
        self.mean_acc: float = 0.
        self.mean_dice: float = 0.
        self.iou: defaultdict = defaultdict(lambda : 0.)
        self.acc: defaultdict = defaultdict(lambda : 0.)
        self.dice: defaultdict = defaultdict(lambda : 0.)
        self.ata: float = 0
        self._counter: int = 0
        self._num_img_with_ata: int = 0
        self._cls_stats: Dict[str, ClassWiseStatistics] = {c: ClassWiseStatistics(c, 0, 0, 0) for c in self.classes}

    def update(self, new_instance: Union[Metric, Dict]):
        if isinstance(new_instance, dict):
            new_instance = Metric(**new_instance)
        self._counter += 1
        self.mean_acc += new_instance.mean_acc
        self.mean_iou += new_instance.mean_iou
        self.mean_dice += new_instance.mean_dice
        for c in self.classes:
            try:
                i, a, d = new_instance.iou[c], new_instance.acc[c], new_instance.dice[c]
                is_valid = all(map(lambda x: ((isinstance(x, torch.Tensor) and not torch.isnan(x)) or not np.isnan(x)) and abs(x) > 1e-3, [i, a, d]))
                if is_valid:
                    self._cls_stats[c].iou += i
                    self._cls_stats[c].acc += a
                    self._cls_stats[c].dice += d
                    self._cls_stats[c].count()
            except KeyError:
                continue

        if new_instance.ata is not None:
            self.ata += new_instance.ata
            self._num_img_with_ata += 1

    @property
    def average(self) -> Metric:
        if self._counter == 0:
            return Metric(0, 0, 0, {}, {}, {})
        mean_acc = self.mean_acc / self._counter
        mean_iou = self.mean_iou / self._counter
        mean_dice = self.mean_dice / self._counter
        iou, dice, acc = {}, {}, {}
        for c in self.classes:
            counter = self._cls_stats[c].counter
            iou[c] = (self._cls_stats[c].iou / counter) if counter != 0 else 0
            acc[c] = (self._cls_stats[c].acc / counter) if counter != 0 else 0
            dice[c] = (self._cls_stats[c].dice / counter) if counter != 0 else 0
        ata = (self.ata / self._num_img_with_ata) if self.ata is not None else None
        return Metric(mean_acc, mean_iou, mean_dice, iou, acc, dice, ata=ata)

    def get_count(self) -> int:
        return self._counter


@dataclass
class StatisticsWithMainObject(Statistics):
    main_object_iou: float = 0
    main_object_acc: float = 0

    def update(self, new_instance: Union[MetricWithMainObject, Dict]):
        if isinstance(new_instance, dict):
            new_instance = MetricWithMainObject(**new_instance)
        super().update(new_instance)
        self.main_object_iou += new_instance.main_object_iou
        self.main_object_acc += new_instance.main_object_acc

    @property
    def average(self) -> MetricWithMainObject:
        base_metric = super().average
        main_object_iou = self.main_object_iou / self._counter
        main_object_acc = self.main_object_acc / self._counter
        return MetricWithMainObject(**asdict(base_metric), main_object_iou=main_object_iou, main_object_acc=main_object_acc)


DictGroupStats = Dict[str, Statistics]
DictGroupMainObjStats = Dict[str, StatisticsWithMainObject]
DictDepthStats = Dict[float, DictGroupStats]


def init_stats(classes: List[str],
               depth_thresholds: List[float],
               stat_groups: List[str],
               with_main_object: bool = False
               ) -> Union[DictGroupStats, DictGroupMainObjStats]:
    if any(map(lambda _dt: not isinstance(_dt, (int, float)) or _dt <= 0, depth_thresholds)):
        raise ValueError("Depth threshold must be a positive number.")

    if not with_main_object:
        stats: DictDepthStats = {}
        for dt in depth_thresholds:
            stat: DictGroupStats = {'all': Statistics(classes=classes)}
            stat.update({g: Statistics(classes=classes) for g in stat_groups})
            stats.update({dt: stat})
    else:
        stats: DictGroupMainObjStats = {'all': StatisticsWithMainObject(classes=classes)}
        stats.update({g: StatisticsWithMainObject(classes=classes) for g in stat_groups})
    return stats


def extract_main_object(
        im: np.ndarray,
        pred: np.ndarray,
        anno: np.ndarray,
        classes: Classes,
        center_class: str = 'trunk',
        expand_ratio_x: float = 0.3,
        expand_ratio_y: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the main object from the image, prediction, and annotation.

    Args:
        im: image
        pred: prediction
        anno: annotation
        classes: classes
        center_class: center class
        expand_ratio_x: expand ratio in x direction
        expand_ratio_y: expand ratio in y direction

    Returns:
        im: image of the main object
        pred: prediction of the main object
        anno: annotation of the main object
        main_object_mask: mask of the main object
    """
    target_mask = (anno == classes[center_class])
    main_object_mask, target_bbox = get_largest_connected_component_in_center(target_mask, img_shape=anno.shape[:2])
    expanded_bbox = expand_bbox(target_bbox,
                                expand_ratio_x=expand_ratio_x,
                                expand_ratio_y=expand_ratio_y,
                                img_shape=anno.shape[:2])
    pred = pred[expanded_bbox.y:expanded_bbox.y + expanded_bbox.h, expanded_bbox.x:expanded_bbox.x + expanded_bbox.w]
    anno = anno[expanded_bbox.y:expanded_bbox.y + expanded_bbox.h, expanded_bbox.x:expanded_bbox.x + expanded_bbox.w]
    im = im[expanded_bbox.y:expanded_bbox.y + expanded_bbox.h, expanded_bbox.x:expanded_bbox.x + expanded_bbox.w]
    main_object_mask = main_object_mask[expanded_bbox.y:expanded_bbox.y + expanded_bbox.h,
                       expanded_bbox.x:expanded_bbox.x + expanded_bbox.w]
    return im, pred, anno, main_object_mask

def _calc_ata(pred: np.ndarray, anno: np.ndarray, valid_anno: np.ndarray,
              trunk_idx: int = 3, canopy_idx: int = 4) -> Optional[float]:
    """
    Calculates adjusted trunk accuracy.

    Assume that predicted trunk pixels may correspond to ground-truth canopy, in addition to ground-truth trunk.

    Define
    - true positive (tp): predicted trunk pixel that corresponds to ground-truth trunk pixel or canopy pixel
    - false positive (fp): predicted trunk pixel that corresponds to ground-truth non-trunk and non-canopy pixel
    - false negative (fn): ground-truth trunk pixel that is not predicted as trunk

    Then, adjusted trunk accuracy(~iou) is defined as tp / (tp + fp + fn).

    Args:
        pred: prediction
        anno: annotation
        valid_anno: mask of valid annotation
        trunk_idx: index of trunk class
        canopy_idx: index of canopy class

    Returns:
        adjusted trunk accuracy
    """
    _pred, _anno = pred[valid_anno], anno[valid_anno]
    trunk_gt = _anno == trunk_idx
    if np.sum(trunk_gt) == 0:
        return None
    tp = np.sum(np.bitwise_and(_pred == trunk_idx, np.bitwise_or(trunk_gt, _anno == canopy_idx)))
    fp = np.sum(np.bitwise_and(_pred == trunk_idx, np.bitwise_and(~trunk_gt, _anno != canopy_idx)))
    fn = np.sum(np.bitwise_and(_pred != trunk_idx, trunk_gt))
    denominator = tp + fp + fn
    return (tp / denominator) if denominator != 0 else None


def _eval_binary(pred: np.ndarray, anno: np.ndarray, valid_anno: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Calculate metrics (iou, acc) for binary segmentation.

    Args:
        pred: prediction
        anno: annotation
        valid_anno: mask of valid annotation

    Returns:
        iou: intersection over union
        acc: accuracy
    """
    if valid_anno is None:
        valid_anno = np.ones_like(anno, dtype=bool)
    _pred, _anno = pred[valid_anno], anno[valid_anno]
    intersection = np.sum(np.bitwise_and(_pred, _anno))
    union = np.sum(np.bitwise_or(_pred, _anno))
    iou = intersection / union if union != 0 else 0
    acc = intersection / np.sum(_anno)
    return iou, acc


def _intersect_and_union(pred_label,
                         label,
                         nclasses,
                         valid_mask: np.ndarray):
    """ Obtained from mmseg.core.evaluation.metrics
    Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map
        label (ndarray): Ground truth segmentation map
        nclasses (int): Number of categories.
        valid_mask (ndarray): valid area mask in evaluation.

     Returns:
        torch.Tensor: The intersection of prediction and ground truth histogram on all classes.
        torch.Tensor: The union of prediction and ground truth histogram on all classes.
        torch.Tensor: The prediction histogram on all classes.
        torch.Tensor: The ground truth histogram on all classes.
        torch.Tensor: The valid area mask on all classes.
    """

    pred_label = torch.from_numpy(pred_label)
    label = torch.from_numpy(label)

    pred_label = pred_label[valid_mask]
    label = label[valid_mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(nclasses), min=0, max=nclasses - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(nclasses), min=0, max=nclasses - 1)
    area_label = torch.histc(
        label.float(), bins=(nclasses), min=0, max=nclasses - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def evaluate_crop(crop: Union[Crop, np.ndarray],
                  annotation: Union[str, np.ndarray],
                  classes: Dict[str, int],
                  cropped_annotation: bool = False,
                  ignored_index: int = 255,
                  ) -> Metric:
    if isinstance(annotation, str):
        annotation = np.load(annotation) if annotation.endswith(".npy") else cv2.imread(annotation, -1)
        cropped_annotation = False
    if isinstance(crop, Crop) and crop.pred is None:
        raise ValueError(f'Crop {crop} has no prediction!')
    if isinstance(crop, Crop):
        if not cropped_annotation:
            annotation = annotation[crop.y:crop.y + crop.h, crop.x:crop.x + crop.w]
        if crop.pred.shape != annotation.shape:
            raise ValueError(f"The shape of the crop's prediction {crop.pred.shape} (corresponding to {crop.img_name})"
                             f"does not match the shape of the annotation {annotation.shape}!")
        pred = crop.pred
    elif isinstance(crop, np.ndarray):
        if not cropped_annotation:
            raise NotImplementedError("This case is not implemented!")
        if crop.shape != annotation.shape:
            raise ValueError(f"The shape of the crop's prediction {crop.shape} does not match "
                             f"the shape of the annotation {annotation.shape}!")
        pred = crop
    else:
        raise TypeError(f'crop must be of type Crop or np.ndarray, not {type(crop)}!')

    return evaluate(pred, annotation, classes, ignored_index=ignored_index)


def evaluate_main_object(pred: np.ndarray,
                         anno: np.ndarray,
                         main_object_mask: np.ndarray,
                         center_class: str,
                         classes: Classes,
                         ignored_classes: Optional[List[str]] = None,
                         ignored_index: int = 255,
                         ) -> MetricWithMainObject:
    """
    Evaluate the main object.

    Args:
        pred: prediction
        anno: annotation
        main_object_mask: mask of the main object
        center_class: center class with concentration
        classes: classes
        ignored_classes: ignored classes
        ignored_index: index of unlabeled class
    """
    crop_metric = evaluate(pred, anno, classes, ignored_classes=ignored_classes, ignored_index=ignored_index)
    center_class = classes[center_class]
    pred_bin = pred == center_class
    non_main_anno_mask = (anno == center_class) & ~main_object_mask
    main_object_metric = _eval_binary(pred_bin, main_object_mask, valid_anno=~non_main_anno_mask)
    return MetricWithMainObject(**asdict(crop_metric),
                                main_object_iou=main_object_metric[0], main_object_acc=main_object_metric[1])


def evaluate(pred: np.ndarray,
             annotation: np.ndarray,
             classes: Classes,
             depth_map: Optional[np.ndarray] = None,
             depth_threshold: Optional[float] = 20.,
             ignored_index: int = 255,
             ignored_classes: Optional[List[str]] = None,
             sky_idx: Optional[int] = 0,
             ) -> Metric:
    """
    Evaluates the prediction against the annotation with optional depth map.

    Args:
        pred: prediction
        annotation: annotation
        classes: classes
        depth_map: depth map (np.ndarray)
        depth_threshold: depth threshold. Ignore the prediction if the corresponding depth is larger than this value.
        ignored_index: ignored index
        ignored_classes: ignored classes
        sky_idx: index of sky class
        exclude_depth_zero: exclude zeros in the depth map

    Returns:
        the metric/evaluation of this image
    """
    if depth_map is not None and depth_threshold > 0:
        if depth_map.shape != annotation.shape:
            depth_map = cv2.resize(depth_map, (annotation.shape[1], annotation.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid_depth_map = ~np.bitwise_and(depth_map == 0, annotation != sky_idx)
        valid_anno = (annotation != ignored_index) & (depth_map <= depth_threshold) & valid_depth_map
    else:
        valid_anno = annotation != ignored_index

    if ignored_classes is not None:
        for c in ignored_classes:
            valid_anno = np.bitwise_and(valid_anno, annotation != classes[c])
    else:
        ignored_classes = []

    area_intersect, area_union, area_pred_label, area_label = _intersect_and_union(pred, annotation, len(classes),
                                                                                   valid_anno)
    cls_iou = area_intersect / area_union
    cls_dice = 2 * area_intersect / (area_pred_label + area_label)
    cls_acc = area_intersect / area_label

    acc = {c: float(cls_acc[i]) for i, c in enumerate(classes) if c not in ignored_classes}
    iou = {c: float(cls_iou[i]) for i, c in enumerate(classes) if c not in ignored_classes}
    dice = {c: float(cls_dice[i]) for i, c in enumerate(classes) if c not in ignored_classes}

    mean_acc = float(np.nanmean(list(acc.values())))
    mean_iou = float(np.nanmean(list(iou.values())))
    mean_dice = float(np.nanmean(list(dice.values())))

    results = {
        'acc': acc,
        'iou': iou,
        'dice': dice,
        'mean_acc': mean_acc,
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
    }

    ata = _calc_ata(pred, annotation, valid_anno, trunk_idx=classes['trunk'], canopy_idx=classes['canopy'])
    if ata is not None:
        results['ata'] = float(ata)

    return Metric(**results)


class EvalProcessor:
    def __init__(self, outputs: Optional[List[str]], show_dir: pathlib.Path,
                 classes: Classes, palettes: Palette,
                 depth_thresholds: List[float] = []):
        self.outputs = outputs
        self.show_dir = show_dir
        self.classes = classes
        self.palettes = palettes
        self.depth_thresholds = depth_thresholds

        if 'comp_img' in outputs:
            self._fig2, self._fig3 = initialize_figures()
            self._piecewise_cmap = get_piecewise_cmap(depth_thresholds) if len(depth_thresholds) > 1 else None
        else:
            self._fig2, self._fig3, self._piecewise_cmap = None, None, None

    def process_whole_image(
            self,
            im: np.ndarray,
            img_name: str,
            pred: np.ndarray,
            anno: np.ndarray,
            conf: Optional[np.ndarray],
            stat: DictDepthStats,
            show_dir: Optional[pathlib.Path] = None,
            depth_map: Optional[np.ndarray] = None,
            ignored_classes: Optional[List[str]] = [],
            opacity: float = 0.8,
            show_ori_img: bool = False,
    ):
        metrics = {}
        for dt in self.depth_thresholds:
            metric = evaluate(pred, anno, self.classes, depth_map=depth_map, depth_threshold=dt,
                              ignored_classes=ignored_classes)
            metrics.update({dt: metric})
            for g in stat[dt].keys():
                if g == 'all' or g in img_name:
                    stat[dt][g].update(metric)
        save_results(self.outputs,
                     show_dirs=show_dir,
                     result=metrics,
                     img=im,
                     img_name=img_name,
                     pred=pred,
                     conf=conf,
                     anno=anno,
                     palettes=self.palettes,
                     depth=depth_map,
                     depth_threshold=self.depth_thresholds,
                     opacity=opacity,
                     fig2=self._fig2,
                     fig3=self._fig3,
                     piecewise_cmap=self._piecewise_cmap,
                     show_im=show_ori_img,
                     )

    def process_crop(
            self,
            im: np.ndarray,
            img_name: str,
            pred: np.ndarray,
            anno: np.ndarray,
            center_class: str,
            expand_ratio_x: float,
            expand_ratio_y: float,
            stat: DictGroupMainObjStats,
            ignored_classes: Optional[List[str]] = [],
            opacity: float = 0.8,
            show_ori_img: bool = False,
    ):
        im, pred, anno, main_object_mask = extract_main_object(im, pred, anno, self.classes,
                                                               center_class,
                                                               expand_ratio_x, expand_ratio_y)
        metric = evaluate_main_object(pred, anno, main_object_mask=main_object_mask, classes=self.classes,
                                      center_class=center_class,
                                      ignored_classes=ignored_classes)
        for g in stat.keys():
            if g == 'all' or g in img_name:
                stat[g].update(metric)
        save_results(self.outputs,
                     show_dirs=self.show_dir,
                     result=metric,
                     img=im,
                     img_name=img_name,
                     pred=pred,
                     conf=None,
                     anno=anno,
                     palettes=self.palettes,
                     opacity=opacity,
                     fig2=self._fig2,
                     fig3=self._fig3,
                     show_im=show_ori_img,
                     )
