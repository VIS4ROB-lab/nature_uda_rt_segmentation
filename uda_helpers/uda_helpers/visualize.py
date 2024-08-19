"""
This module contains functions for visualizing the semantic segmentation results, depth map, and evaluation results.

For visualize the semantic overlaid images, call `visualize_sem_with_img`.
To visualize semantic maps with command line, call vis instead of this script.
"""

# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Union, List, Dict, Optional, Tuple
import pathlib
import warnings
from itertools import cycle

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

from .types import Palette
from .utils import Metric, MetricWithMainObject, extract_filestem


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
          (200, 153, 255), (189, 246, 252), (222, 244, 208), (249, 222, 169), (249, 193, 228),
          (251, 248, 204), (253, 228, 207), (255, 207, 210), (241, 192, 232), (207, 186, 240),
          (163, 196, 243), (144, 219, 244), (142, 236, 245), (152, 245, 225), (185, 251, 192)
          ]
colors = cycle(colors)


def _convert(sem_seg: np.ndarray, palette: np.ndarray, rgb2bgr: bool) -> np.ndarray:
    _color_seg = np.zeros((sem_seg.shape[0], sem_seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        _color_seg[sem_seg == label, :] = color
    if rgb2bgr:
        _color_seg = _color_seg[..., ::-1]  # RGB -> BGR
    return _color_seg


def visualize_sem_with_img(
        img: Union[str, pathlib.Path, np.ndarray, None],
        sem_seg: Union[np.ndarray, str],
        palette: Palette,
        opacity: float = 0.5, rgb2bgr=True) -> np.ndarray:
    """
    Generate the colored images with the semantic segmentation results overlaid on the original image.

    Args:
        img (str | pathlib.Path | np.ndarray): Path to the original image or the original image.
        sem_seg (np.ndarray | str): Semantic segmentation results.
        palette (Palette): Color palette.
        opacity (float): Opacity of the overlaid segmentation results.

    Returns:
        np.ndarray: The colored overlaid image.
    """

    palette = np.array(list(palette.values()))

    if img is not None:
        if isinstance(img, str):
            img = cv2.imread(img, -1)
        elif isinstance(img, pathlib.Path):
            img = cv2.imread(img.as_posix(), -1)
        if not isinstance(img, np.ndarray):
            raise FileNotFoundError(f"File {img} does not exist!")
        if isinstance(sem_seg, str):
            sem_seg = cv2.imread(sem_seg, -1)
        if sem_seg is None or not isinstance(sem_seg, np.ndarray):
            raise ValueError(f"Invalid semantic segmentation results!")
        if sem_seg.shape[:2] != img.shape[:2]:
            warnings.warn(f"Shape of the semantic segmentation results {sem_seg.shape[:2]} does not match the shape of "
                          f"the image {img.shape[:2]}!", category=RuntimeWarning)
            sem_seg = cv2.resize(sem_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        color_seg = _convert(sem_seg, palette, rgb2bgr)

        return (img * (1 - opacity) + color_seg * opacity).astype(np.uint8)
    else:
        return _convert(sem_seg, palette, rgb2bgr)


def visualize_depth_map(depth_map: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    if isinstance(threshold, float):
        within_threshold = depth_map <= threshold
        clipped_depth_map = np.clip(depth_map, 0, threshold).astype(np.half) / threshold
        rgb_image = np.where(within_threshold[..., np.newaxis], plt.cm.BuGn(clipped_depth_map)[:,:,:3], 0)
    else:
        rgb_image = plt.cm.BuGn(depth_map)[:,:,:3]

    return rgb_image


def initialize_figures():
    fig_2_subplots, _ = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    fig_3_subplots, _ = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    return fig_2_subplots, fig_3_subplots


def clear_axes(fig):
    for ax in fig.axes:
        ax.clear()


def get_piecewise_cmap(thresholds: List[float], existing_cmap: str = 'BuGn') -> LinearSegmentedColormap:
    try:
        existing_cmap = matplotlib.colormaps.get_cmap(existing_cmap)
    except ValueError:
        existing_cmap = matplotlib.colormaps.get_cmap('gray')

    norm_min, norm_max = min(thresholds), max(thresholds)
    normalized_thresholds = (np.array(thresholds) - norm_min) / (norm_max - norm_min)

    # Ensure the first and last elements are 0 and 1, respectively
    if normalized_thresholds[0] != 0:
        normalized_thresholds = np.insert(normalized_thresholds, 0, 0)
    if normalized_thresholds[-1] != 1:
        normalized_thresholds = np.append(normalized_thresholds, 1)

    norm = plt.Normalize(0, 1)
    colors = [existing_cmap(norm(value)) for value in normalized_thresholds]
    piecewise_cmap = LinearSegmentedColormap.from_list("DepthCMap", list(zip(normalized_thresholds, colors)))

    return piecewise_cmap


def save_results(
        outputs: Optional[List[str]],
        show_dirs: pathlib.Path,
        result: Union[Metric, Dict[float, Metric], MetricWithMainObject, None],
        img: Optional[Union[str, pathlib.Path, np.ndarray, None]],
        img_name: Optional[str],
        pred: Union[np.ndarray, str],
        conf: Optional[np.ndarray],
        anno: Optional[np.ndarray],
        palettes: Palette,
        depth: Optional[np.ndarray] = None,
        depth_threshold: Optional[Union[float, List[float]]] = 20,
        opacity: float = 0.5,
        fig2: Optional[plt.Figure] = None,
        fig3: Optional[plt.Figure] = None,
        show_im: bool = False,
        piecewise_cmap = None
) -> None:
    """
    Save the inference / evaluation results as images, compared images or numpy arrays.

    Args:
        outputs (Optional[List[str]]): List of outputs to be saved. If None, no output will be saved.
        show_dirs (pathlib.Path): Path to the directory where the outputs will be saved.
        result (Union[Metric, Dict[float, Metric], MetricWithMainObject, None]): Evaluation results, in dict for different depths or a single Metric.
        img (Optional[Union[str, pathlib.Path, np.ndarray, None]]): Path to the original image or the original image.
        img_name (Optional[str]): Name of the original image.
        pred (Union[np.ndarray, str]): Semantic segmentation results.
        conf (Optional[np.ndarray]): Logits from the model/Confidence map.
        anno (Optional[np.ndarray]): Annotation.
        depth (Optional[np.ndarray]): Depth map (if available), cut at the threshold.
        depth_threshold (float | List[float] | None): Threshold for the depth map.
        palettes (Palette): Color palette.
        opacity (float): Opacity of the overlaid segmentation results.
        fig2 (Optional[plt.Figure]): Figure with 2 subplots.
        fig3 (Optional[plt.Figure]): Figure with 3 subplots.
        show_im (bool): Whether to show the image when there is no depth map.
        piecewise_cmap (Optional): Piecewise color map for the depth map.
    Returns:
        None
    """
    if outputs is not None:

        img_name = img.name if isinstance(img, pathlib.Path) else img if isinstance(img, str) else img_name

        if "img" in outputs:
            overlaid_img = visualize_sem_with_img(img, sem_seg=pred, palette=palettes, opacity=opacity)
            save_path = show_dirs / img_name
            cv2.imwrite(save_path.as_posix(), overlaid_img)
        if "comp_img" in outputs:
            if result is None:
                raise ValueError(f"Result is None for image {img_name}")
            overlaid_img = visualize_sem_with_img(img, pred, palettes, opacity=opacity)
            overlaid_anno = visualize_sem_with_img(img, anno, palettes, opacity=opacity)
            num_subplots = 2 if (depth is None) and (not show_im) else 3

            fig = fig2 if num_subplots == 2 else fig3
            clear_axes(fig)
            axes = fig.axes

            # fig, axes = plt.subplots(1, num_subplots, figsize=(int(6*num_subplots), 6), constrained_layout=True)
            axes[0].imshow(overlaid_img[..., ::-1])
            axes[1].imshow(overlaid_anno[..., ::-1])
            axes[0].set_title("Prediction")
            axes[1].set_title("Annotation")
            if depth is not None and depth_threshold is not None:
                if isinstance(depth_threshold, float):
                    axes[2].imshow(visualize_depth_map(depth, depth_threshold,))
                    axes[2].set_title(f"Depth map at threshold {depth_threshold}")
                elif len(depth_threshold) == 1 and isinstance(depth_threshold[0], (float, int)):
                    axes[2].imshow(visualize_depth_map(depth, depth_threshold[0]))
                    axes[2].set_title(f"Depth map at threshold {depth_threshold[0]}")
                else:
                    # dmap = axes[2].imshow(depth, cmap=piecewise_cmap if piecewise_cmap is not None else 'BuGn',
                    #                       vmin=0, vmax=max(depth_threshold))
                    dmap = axes[2].imshow(visualize_depth_map(depth, max(depth_threshold)))
                    axes[2].set_title("Depth map")
                    # fig.colorbar(dmap, ax=axes[2], orientation='vertical')
            elif show_im:
                axes[2].imshow(img[..., ::-1])
                axes[2].set_title("Image")
            for ax in axes:
                ax.axis("off")
            title = ""
            if isinstance(result, (Metric, MetricWithMainObject)):
                title = f"mIoU: {result.mean_iou:.4f}, mAcc: {result.mean_acc:.4f}"
                if result.acc.get("canopy", 0):
                    title += f", Canopy Acc: {result.acc['canopy']:.4f}"
                if result.ata is not None:
                    title += f", adjusted trunk acc: {result.ata:.4f}"
                if isinstance(result, MetricWithMainObject):
                    title += f"\nMain object IoU: {result.main_object_iou:.4f}, acc: {result.main_object_acc:.4f}"
            elif isinstance(result, dict):
                for d, m in result.items():
                    title_line = f"Depth: {d}, mIoU: {m.mean_iou:.4f}, mAcc: {m.mean_acc:.4f}"
                    if m.acc.get("canopy", 0):
                        title_line += f", Canopy Acc: {m.acc['canopy']:.4f}"
                    if m.ata is not None:
                        title_line += f", adjusted trunk acc: {m.ata:.4f}"
                    title += title_line + "\n"
            fig.suptitle(title)
            # plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
            save_path = show_dirs / (extract_filestem(img_name) + '_comp.png')
            fig.savefig(save_path.as_posix(), dpi=300)
            # plt.close(fig)
        if isinstance(pred, str):
            pred = cv2.imread(pred, -1)
        if "png" in outputs:
            save_path = show_dirs / (extract_filestem(img_name) + '.png')
            cv2.imwrite(save_path.as_posix(), pred)
        if "npy" in outputs:
            save_path = show_dirs / (extract_filestem(img_name) + '.npy')
            np.save(save_path.as_posix(), pred.astype(np.uint8))
        if "conf" in outputs:
            conf_path = show_dirs / (extract_filestem(img_name) + '_conf.npy')
            np.save(conf_path.as_posix(), conf.astype(np.half))


def plot_distribution(distributions: Union[Dict[str, List[float]], List[Dict[str, List[float]]]], save_path: Union[str, pathlib.Path], series: Optional[List[str]] = None) -> None:
    sns.set(style="darkgrid")
    if series is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        if not isinstance(distributions, dict):
            raise RuntimeError("Invalid distribution data type! Expecting Dict[str, List[float]]")
        try:
            sns.histplot(data=distributions['mIoU'], ax=axes[0, 0], kde=False)
            axes[0, 0].set_title("mIoU")

            sns.histplot(data=distributions['mAcc'], ax=axes[0, 1], kde=False)
            axes[0, 1].set_title("mAcc")

            sns.histplot(data=distributions['cc'], ax=axes[1, 0], kde=False)
            axes[1, 0].set_title("Canopy Coverage")

            sns.histplot(data=distributions['ata'], ax=axes[1, 1], kde=False)
            axes[1, 1].set_title("Adjusted Trunk Accuracy")
        except KeyError as e:
            raise RuntimeError("Invalid distribution data with key error:", e)
    else:
        fig, axes = plt.subplots(len(series), 2, figsize=(12, 4*len(series)))
        palette = sns.color_palette(n_colors=len(distributions))
        if not isinstance(distributions, list):
            raise RuntimeError("Invalid distribution data type! Expecting List[Dict[str, List[float]]]")
        for i, d in enumerate(distributions):
            color = palette[i]

            axes[i, 0].set_title("mIoU")
            sns.histplot(x=d['mIoU'], ax=axes[i, 0], kde=False, label=series[i], color=color)

            axes[i, 1].set_title("Main Object IoU")
            sns.histplot(x=d['main_object_IoU'], ax=axes[i, 1], kde=False, label=series[i], color=color)

        fig.legend()
        fig.suptitle("Distributions of mIoU and Main Object IoU")

    plt.savefig(save_path)
    plt.close(fig)


def annotate_points(bg: np.ndarray, points: Union[List[Tuple], np.ndarray], color: Optional[Tuple[int, int, int]] = (255, 255, 0),
                    radius: int = 15):
    if len(bg.shape) == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    if color is None:
        color = next(colors)
    for p in points:
        cv2.circle(bg, (p[0], p[1]), radius, color, -1)
    return bg
