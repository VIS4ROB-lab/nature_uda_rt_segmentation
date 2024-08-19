# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Tuple, Optional, Union, Callable, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import random
from copy import deepcopy

import cv2
import numpy as np


class RefineType(Enum):
    REGION = 'region'
    BOUNDARY_SHARPENING = 'boundary-sharpening'
    LABEL_RECOVERY = 'label-recovery'
    REMOVE_HOLES = 'remove_holes'


class PostProcessorType(Enum):
    GLOBAL = 1
    INSTANCE = 2


class VerboseLevel(Enum):
    """
    Verbose level for the refine task.

    NONE: No verbose.
    TASK: Save results for each task.
    ALL: Save results at intermediate steps.
    """
    NONE = 0
    TASK = 1
    ALL = 2


@dataclass
class RefinedInstance:
    img: Optional[np.ndarray] = None
    pred: Optional[np.ndarray] = None
    new_pred: Optional[np.ndarray] = None


@dataclass
class ErosionConfig:
    erode_kernel_size: int
    erode_iterations: int
    dilation_kernel_size: int = 3
    dilation_iterations: int = 1

    def __post_init__(self):
        attrs_to_check = [
            self.erode_kernel_size,
            self.erode_iterations,
            self.dilation_kernel_size,
            self.dilation_iterations
        ]
        for attr in attrs_to_check:
            if not isinstance(attr, int) or attr < 0:
                raise ValueError(f"Expected a non-negative integer, but got {attr}")

    def as_dict(self):
        return asdict(self)


PostProcessorDict = Dict[PostProcessorType, Tuple[Callable, Dict[str, Any]]]


@dataclass
class RefineTask:
    type: RefineType
    foreground_id: Optional[int]
    negative_cls_id: Optional[Union[int, List[int]]] = None
    num_sample_points: int = 6
    min_valid_area: Optional[int] = None
    erosion_cfg: Optional[ErosionConfig] = None
    post_processor: Optional[PostProcessorDict] = None
    ignore_top_y: Optional[Union[float, int]] = None
    allowed_aspect_ratio: Optional[Tuple[float, float]] = None
    max_expansion_ratio: Optional[float] = 150
    allowed_connected_components: int = -1

    def __post_init__(self):
        if not isinstance(self.type, RefineType):
            raise ValueError(f"Invalid refine type: {self.type}")
        if self.allowed_connected_components < -1 or self.allowed_connected_components == 0:
            raise ValueError(f"Invalid allowed_connected_components: {self.allowed_connected_components}")
        if isinstance(self.erosion_cfg, dict):
            try:
                self.erosion_cfg = ErosionConfig(**self.erosion_cfg)
            except TypeError:
                raise ValueError("Invalid erosion_cfg!")

    def __str__(self):
        out = "\n------------\n"
        out += f"RefineType: {self.type}, foreground: {self.foreground_id},\n"
        if self.negative_cls_id is not None:
            out += f"negative class: {self.negative_cls_id}, "
        out += f"num sample points: {self.num_sample_points}, "
        if self.min_valid_area is not None:
            out += f"min valid area: {self.min_valid_area}, "
        if self.erosion_cfg is not None:
            out += f"erosion config: {self.erosion_cfg.as_dict()}, "
        if self.post_processor is not None:
            out += f"post processors: {self.post_processor.keys()}, "
        if self.ignore_top_y is not None:
            out += f"ignore top y: {self.ignore_top_y}, "
        if self.allowed_aspect_ratio is not None:
            out += f"allowed aspect ratio: {self.allowed_aspect_ratio}, "
        if self.max_expansion_ratio is not None:
            out += f"max expansion ratio: {self.max_expansion_ratio}, "
        if self.allowed_connected_components != -1:
            out += f"allowed connected components: {self.allowed_connected_components}, "
        out += "\n------------\n"
        return out

    def short_repr(self):
        out = f"\nType: {self.type}, foreground: {self.foreground_id}, "
        if self.negative_cls_id is not None:
            out += f"negative class: {self.negative_cls_id}, "
        out += f"num sample points: {self.num_sample_points}, "
        out += "\n"
        return out


def _halton_generator(base):
    """Generator function for Halton sequence for a given base."""
    index = 1
    while True:
        result = 0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f = f / base
        yield result
        index += 1


def _xywh_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get the bounding box of a boolean mask.
    """
    rows, cols = np.where(mask)
    x, y = np.min(cols), np.min(rows)
    w, h = np.max(cols) - x + 1, np.max(rows) - y + 1
    return x, y, w, h


def filter_aspect_ratio(bin_map: np.ndarray,
                        min_aspect_ratio: float = 0.0,
                        max_aspect_ratio: float = 1.0) -> np.ndarray:
    """
    Filter out connected components with aspect ratio outside the given range.

    Args:
        bin_map (np.ndarray): The binary map.
        min_aspect_ratio (float): The minimum aspect ratio. Defaults to 0.
        max_aspect_ratio (float): The maximum aspect ratio. Defaults to 1.

    Returns:
        np.ndarray: The filtered binary map. In boolean if the input is boolean.
    """
    bool_flag = False
    if bin_map.dtype == bool:
        bool_flag = True
        bin_map = bin_map.astype(np.uint8) * 255
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bin_map, 8)
    stats = stats[1:]  # Skip the background label 0
    filtered = np.zeros_like(bin_map)
    for i in range(len(stats)):
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if min_aspect_ratio <= w / h <= max_aspect_ratio:
            filtered[labels == (i + 1)] = 1
    return filtered.astype(bool) if bool_flag else filtered


def sample_2d_from_region(region: np.ndarray, num_sample_points: int = 6, sampling: str = 'halton',
                          **kwargs) -> List[Tuple]:
    """
    Sample 2D points from a region using Halton sequence or randomly.
    """
    if np.all(region == 0) or np.all(region is False):
        return []
    _x, _y, _w, _h = _xywh_from_mask(region)
    x, y, w, h = kwargs.get('x', _x), kwargs.get('y', _y), kwargs.get('w', _w), kwargs.get('h', _h)
    samples = []
    if sampling == 'halton':
        halton_2 = _halton_generator(2)
        halton_3 = _halton_generator(3)
        while len(samples) < num_sample_points:
            hx, hy = next(halton_2), next(halton_3)
            sx, sy = int(x + hx * w), int(y + hy * h)
            if region[sy, sx]:
                samples.append((sx, sy))
    elif sampling == 'random':
        while len(samples) < num_sample_points:
            sx, sy = random.randint(x, x + w - 1), random.randint(y, y + h - 1)
            if region[sy, sx]:
                samples.append((sx, sy))
    return samples


def split_instances(bin_map: np.ndarray,
                    min_area: int = 0, max_area: int = -1,
                    num_sample_points: int = 6,
                    ) -> Tuple[List[List[Tuple]], List[np.ndarray]]:
    """
    Split a binary map into instances.

    Args:
        bin_map (np.ndarray): The binary map.
        min_area (int): The minimum area of an instance. Defaults to 0.
        max_area (int): The maximum area of an instance. Defaults to -1 (no limit).
        num_sample_points (int): The number of sample points for each instance, if return points. Defaults to 6.

    Returns:
        Tuple[List[List[Tuple]], List[np.ndarray]]: Instances (samples points) and masks.
    """
    bin_map = bin_map.astype(np.uint8)
    if max_area == -1:
        max_area = bin_map.shape[0] * bin_map.shape[1]

    _, binary_mask = cv2.threshold(bin_map, 0, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8, cv2.CV_32S)

    instances = []
    masks = []
    stats = stats[1:]  # Skip the background label 0
    area_descending_order = stats[:, -1].argsort()[::-1]
    stats = stats[area_descending_order]  # Sort by area in descending order

    for i in range(num_labels - 1):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            samples = sample_2d_from_region(bin_map, x=x, y=y, w=w, h=h, num_sample_points=num_sample_points)
            instances.append(samples)
            masks.append(labels == (area_descending_order[i] + 1))

    return instances, masks


def shrink(bin_map: np.ndarray, erode_kernel_size: int = 5, erode_iterations: int = 2,
           dilation_kernel_size: int = 3, dilation_iterations: int = 1) -> np.ndarray:
    """
    Erode a binary map.

    Args:
        bin_map (np.ndarray): The binary map.
        erode_kernel_size (int): The kernel size of the erosion. Defaults to 5.
        erode_iterations (int): The number of iterations of erosion. Defaults to 2.
        dilation_kernel_size (int): The kernel size of the dilation. Defaults to 3.
        dilation_iterations (int): The number of iterations of dilation. Defaults to 1.

    Returns:
        np.ndarray: The eroded binary map, np.uint8.
    """
    if bin_map.dtype == bool:
        bin_map = bin_map.astype(np.uint8) * 255
    erode_kernel = np.ones((erode_kernel_size, erode_iterations), np.uint8)
    result = cv2.erode(bin_map, erode_kernel, iterations=erode_iterations)
    if dilation_iterations > 0:
        dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        result = cv2.dilate(result, dilation_kernel, iterations=dilation_iterations)
    return result


def pop_top_n_largest_components(n: int,
                                 bin_map: Optional[np.ndarray] = None,
                                 labels: Optional[np.ndarray] = None,
                                 stats: Optional[np.ndarray] = None,
                                 keep_largest: bool = False,
                                 ) -> np.ndarray:
    """
    Pop the mask combining the top N largest connected components.

    Args:
        n (int): The number of components to pop.
        bin_map (np.ndarray): The binary map. Defaults to None. Has to be set if labels and stats are not provided.
        labels (np.ndarray): The connected components labels. Defaults to None. For pre-computed labels, pass this argument.
        stats (np.ndarray): The connected components stats. Defaults to None. For pre-computed stats, pass this argument.
        keep_largest (bool): Whether to keep the largest component. Defaults to False. Set to False if the largest component is the background.
    """
    stats = deepcopy(stats)
    if labels is None or stats is None:
        if bin_map is None:
            raise ValueError("Either bin_map or labels and stats have to be provided!")
        else:
            bin_map = bin_map.astype(np.uint8) * 255
        _, labels, stats, _ = cv2.connectedComponentsWithStats(bin_map, 8)
    if not keep_largest:
        stats = stats[1:]
    area_descending_order = stats[:, -1].argsort()[::-1]
    if n != -1:
        area_descending_order = area_descending_order[:n]
    comp_mask = np.isin(labels, area_descending_order+1) if not keep_largest else np.isin(labels, area_descending_order)
    return comp_mask


def select_mask(masks: np.ndarray,
                neg_mask: np.ndarray,
                allowed_connected_components: int = -1,
                cov_threshold: float = 0.5,
                ) -> Tuple[int, np.ndarray]:
    """
    Select a mask from N masks, according to the coverage threshold.

    If cov_threshold > 0, the mask with the min number of connected component and the lowest coverage will be selected.
    However, if the best coverage is still larger than cov_threshold, the mask with the largest connected component.
    **If cov_threshold is set to 0, the largest connected component across masks will be selected directly.**
    (for boundary sharpening)


    Args:
        masks (np.ndarray): The masks to select from. Shape: (N, H, W).
        neg_mask (np.ndarray): The negative mask. Shape: (H, W).
        allowed_connected_components (int): The maximum number of connected components. Defaults to -1 (no limit).
        cov_threshold (float): The coverage threshold to select the largest component. Defaults to 0.5.

    Returns:
        int: The index of the selected mask.
        Optional[np.ndarray]: The mask of the selected connected components.
    """
    if len(masks.shape) != 3:
        raise ValueError(f"Only support multiple masks! Expected shape: (N, H, W), given {masks.shape}")
    if masks.dtype == bool:
        masks = masks.astype(np.uint8) * 255
    best_idx, best_ncc, best_cov = -1, np.inf, 1.
    temp, comp_mask = None, None
    connectivity_temp = {}
    for i, m in enumerate(masks):
        if cov_threshold < 1e-3:
            break
        if (m_count := np.count_nonzero(m)) == 0:
            continue
        num_connected_components, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        connectivity_temp[i] = (num_connected_components, labels, stats, None)
        if num_connected_components < best_ncc:
            best_idx, best_ncc = i, num_connected_components
            temp = (labels, stats)
            cov = np.count_nonzero(m & neg_mask) / m_count
            best_cov = cov
        elif num_connected_components == best_ncc:
            cov = np.count_nonzero(m & neg_mask) / m_count
            if cov < best_cov:
                best_idx, best_cov = i, cov
            temp = (labels, stats)
    if best_cov < cov_threshold and best_ncc != np.inf:
        if allowed_connected_components != -1 and best_ncc > allowed_connected_components:
            if temp is None:
                raise ValueError("No mask selected!")
            labels, stats = temp
            comp_mask = pop_top_n_largest_components(
                n=allowed_connected_components,
                bin_map=None,
                labels=labels,
                stats=stats,
                keep_largest=False
            )
        return best_idx, comp_mask
    else:
        best_idx, best_area, _ncc = -1, 0, -1
        comp_mask = None
        for i, m in enumerate(masks):
            num_connected_components, labels, stats, _ = connectivity_temp.get(i, cv2.connectedComponentsWithStats(m, 8))
            if num_connected_components == 1:
                continue
            stats = stats[1:]
            area_descending_order = stats[:, -1].argsort()[::-1]
            stats = stats[area_descending_order]
            if stats[0, -1] > best_area:
                best_idx, best_area = i, stats[0, -1]
                _ncc = num_connected_components
                comp_mask = labels == (area_descending_order[0] + 1)
        return best_idx, comp_mask


def calc_continuous_width(arr: Union[np.ndarray, List[bool]]) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Calculate the maximum continuous width of a binary array.

    Args:
        arr (Union[np.ndarray[bool], List[bool]]): The binary array.

    Returns:
        Tuple[int, List[Tuple[int, int]]]: The maximum continuous width and the list of continuous sequences,
        arranged in (start, length).
    """
    true_sequences = []
    current_start = -1

    for i, value in enumerate(arr):
        if value:
            if current_start == -1:
                current_start = i
        else:
            if current_start != -1:
                true_sequences.append((current_start, i - current_start))
                current_start = -1

    # Handle the case where the array ends with a True
    if current_start != -1:
        true_sequences.append((current_start, len(arr) - current_start))

    max_lengths = max([length for _, length in true_sequences], default=0)
    return max_lengths, true_sequences


def check_new_segments(base_segments: List[Tuple[int, int]], new_segments: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """
    Check if there is a new sequence in the segments list.

    Args:
        base_segments (List[Tuple[int, int]]): The base sequence segments.
        new_segments (List[Tuple[int, int]]): The new sequence segments.

    Returns:
        List[Tuple[int, int, int]]: The uncovered segments, arranged in (start, length, distance).
    """
    uncovered_segments = []

    for new_start, new_length in new_segments:
        new_end = new_start + new_length
        is_covered = False
        min_distance = float('inf')

        for base_start, base_length in base_segments:
            base_end = base_start + base_length

            # Check if the new segment is covered by the base segment
            if not (new_end <= base_start or new_start >= base_end):
                is_covered = True
                break
            else:
                # Calculate the distance to the base segment
                distance = min(abs(new_start - base_end), abs(new_end - base_start))
                min_distance = min(min_distance, distance)

        if not is_covered:
            uncovered_segments.append((new_start, new_length, min_distance))

    return uncovered_segments


def yshape_detector(mask: np.ndarray, width_change_threshold: Union[int, float] = 0.2,
                    strict: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect Y-shape/T-shape in a mask according to the threshold of width change. Return both parts.

    Args:
        mask (np.ndarray): The mask to detect. Shape: (H, W).
        width_change_threshold (int|float): The threshold of width change. Integer for absolute value, float for relative value. Defaults to 0.5.
        strict (bool): Whether to use strict mode where new sequence segments trigger directly. Defaults to False.

    Returns:
        np.ndarray: The binary split mask for the rectangle part. Shape: (H, W).
        np.ndarray: The binary split mask for the Y-shape/T-shape top part. Shape: (H, W).
    """
    if len(mask.shape) != 2:
        raise ValueError(f"Only support single mask! Expected shape: (H, W), given {mask.shape}")
    if not isinstance(width_change_threshold, (int, float)):
        raise ValueError(f"width_change_threshold has to be int or float, given {type(width_change_threshold)}")

    shape_x, shape_y, shape_w, shape_h = _xywh_from_mask(mask)
    rows, cols = shape_h, shape_w
    transition_row, running_sum, counter = -1, 0, 1
    base_width, base_height = 0, max(int(0.2 * rows), 10)
    last_sequences = []

    for r in range(shape_y + shape_h - 1, shape_y, -1):
        counter += 1
        # width = np.argmax(mask[r, shape_x:shape_w+shape_x+1]) - np.argmin(mask[r, shape_x:shape_w+shape_x+1])
        width, sequences = calc_continuous_width(mask[r, shape_x:shape_w+shape_x+1].astype(bool))

        if counter < base_height:
            running_sum += width
        elif counter == base_height:
            base_width = (running_sum + width) / counter
            last_sequences = sequences
        else:
            if isinstance(width_change_threshold, float):
                t = width_change_threshold * base_width
            else:
                t = width_change_threshold
            sequence_indicator = False
            new_segments = check_new_segments(last_sequences, sequences)
            if (len(new_segments) and any([(0.25*base_width < length < 2*base_width and 0.25*base_width < distance) for _, length, distance in new_segments])) or (strict and len(new_segments)):
                sequence_indicator = True
            if abs(width - base_width) > t or sequence_indicator:
                transition_row = r + 1
                break
    mask = mask.astype(bool, copy=True)
    rect_mask, yshape_mask = np.zeros_like(mask, dtype=bool), np.zeros_like(mask, dtype=bool)
    if transition_row != -1:
        yshape_mask[shape_y:transition_row, shape_x:shape_x+shape_w] = np.where(mask[shape_y:transition_row, shape_x:shape_x+shape_w], True, False)
        rect_mask[transition_row:, shape_x:shape_x+shape_w] = np.where(mask[transition_row:, shape_x:shape_x+shape_w], True, False)
    else:
        rect_mask = mask.copy()
    return rect_mask, yshape_mask


def yshape_post_processor(mask: np.ndarray, width_change_threshold: int = 10, strict: bool = False) -> None:
    """
    Post-process a mask with Y-shape/T-shape. Extract the rectangle part.
    """
    rect, y = yshape_detector(mask, width_change_threshold=width_change_threshold, strict=strict)
    mask[y] = 0

