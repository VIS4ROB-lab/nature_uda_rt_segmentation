# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import pathlib
from typing import Union, Tuple, Sequence, Optional, Dict, List
from random import choice, choices
import time
import math

import numpy as np
import mmcv


class CopyPasteAug(object):
    """Copy and Paste

    Copy and paste implementation for image augmentations.

    Application may contain (but not limited to) adding special crops to images, such as introducing a specific class
    to a dataset where no instances occurred normally.

    Args:
        crop_source (str or pathlib.Path): Path to source image path.
        crop_extension (str): Extension of crop source images.
        crop_label (int): Target label (index) of the crop to be added on the semantic map. -1 for not modifying the semantic map
        max_pastes (int): Max crop pastes on one image.
        hratio_range (tuple[float, float]): Vertical ratio range of the crop with respect to the image. (in HEIGHT direction)
        wratio_allowance (tuple[float, float]): Allowed horizontal ratio of the crop (in WIDTH direction)
        prob (float): The probability of applying this operation.
        hflip_prob (float): The probability of flipping the crop before adding to the source image.
        keep_asp_ratio_prob (float): Probability of keeping aspect ratio of the pasted crop.
        over_layer_indices (sequence[int], optional): The indices to be placed under the crop layer.
        In other words, the pixels from the images over these class indices may be overwritten by the crop.
        None means only place the layer over the ignored index 255.
        sky_index (int): index of the sky part in the annotation.
        activation_iter (int): The iteration (after a certain num of images to be loaded by the worker)
        to activate this augmentation.
    """

    def __init__(self,
                 crop_source: Union[str, pathlib.Path],
                 crop_extension: str = 'png',
                 crop_label: int = 5,
                 max_pastes: int = 2,
                 hratio_range: Tuple[float, float] = (0.12, 0.2),
                 wratio_allowance: Tuple[float, float] = (0.2, 0.8),
                 prob: float = 0.5,
                 hflip_prob: float = 0.5,
                 keep_asp_ratio_prob: float = 0.5,
                 over_layer_indices: Optional[Sequence[int]] = None,
                 sky_index: int = 0,
                 activation_iter: int = 5000,
                 ):
        crop_source = crop_source if isinstance(crop_source, pathlib.Path) else pathlib.Path(crop_source)
        self.crop_src: List[pathlib.Path] = [f for f in crop_source.rglob("*." + crop_extension)]
        self.crop_src_len = len(self.crop_src)
        if len(hratio_range) != 2 or hratio_range[0] > hratio_range[1] or hratio_range[0] < 0:
            raise ValueError(f"Invalid vertical ratio range given: {hratio_range}")
        self.crop_label = crop_label
        self.max_pastes = max_pastes
        self.hratio_range = hratio_range
        self.wratio_allowance = wratio_allowance
        self.prob = prob
        self.hflip_prob = hflip_prob
        self.keep_asp_ratio_prob = keep_asp_ratio_prob
        self.over_layer_indices = np.array(list(over_layer_indices) + [255]) if over_layer_indices is not None \
            else np.array([255])
        self.activation_iter = int(activation_iter) if activation_iter is not None else 0
        self.sky_index = int(sky_index)

        self.num_pastes_prob_lut = self._get_num_pastes_prob_lut()

        self._max_retries_reading_iter = 5
        self._retry_delay_reading_iter = 0.1

    @staticmethod
    def _sample_2d(w_min: float, w_max: float,
                   h_min: float, h_max: float,
                   paste_num: int,
                   total_paste: int,
                   h_horizon: float = 0.6, ) -> Tuple[float, float]:
        w_diff = w_max - w_min
        w_itvl = w_diff / (total_paste + 1)
        x = np.random.uniform((paste_num - 1) * w_itvl, (paste_num + 1) * w_itvl) + w_min
        if (r := np.random.uniform()) < 0.85:
            y = np.random.uniform(h_horizon * 0.8, h_horizon * 1.05)
        elif 0.85 <= r:
            y = np.random.uniform(h_horizon * 1.05, min(h_max, h_horizon * 1.25))
        # else:
        #     y = np.random.uniform(h_min, h_horizon * 0.8)
        y = float(np.clip(y, h_min, h_max))
        return x, y

    @staticmethod
    def _check_sem_max_y(sem: np.ndarray, target: int) -> int:
        mask = sem == target
        return np.where(mask)[0].max() if np.any(mask) else -1

    @staticmethod
    def _get_current_iter(max_retries_reading_iter: int = 5, retry_delay_reading_iter: float = 0.1) -> int:
        current_iter = 0
        for _ in range(max_retries_reading_iter):
            try:
                with open('.iter.txt', 'r') as f:
                    current_iter = int(f.read().strip())
                break
            except (FileNotFoundError, ValueError):
                time.sleep(retry_delay_reading_iter)
        return current_iter

    def _get_num_pastes_prob_lut(self) -> Dict[int, float]:
        return {
            k: math.comb(self.max_pastes, k) * (self.prob ** k) * ((1 - self.prob) ** (self.max_pastes - k)) for k in
            range(self.max_pastes + 1)} if self.max_pastes > 0 else {0: 1.0}

    def __call__(self, results: Dict) -> Dict:
        iter_count = self._get_current_iter(self._max_retries_reading_iter, self._retry_delay_reading_iter)
        if iter_count < self.activation_iter:
            return results
        img, sem = results["img"], results["gt_semantic_seg"]
        ih, iw = img.shape[:2]
        max_sky_y = self._check_sem_max_y(sem, self.sky_index)
        if max_sky_y == -1:
            return results
        max_sky_y = max_sky_y / ih

        num_pastes = choices(list(self.num_pastes_prob_lut.keys()), weights=list(self.num_pastes_prob_lut.values()))[0]

        for i in range(1, num_pastes + 1):
            crop_img = mmcv.imread(choice(self.crop_src).as_posix(), 'unchanged')
            if np.random.uniform() < self.hflip_prob:
                crop_img = mmcv.imflip(crop_img, 'horizontal')
            crop_height = np.random.uniform(*self.hratio_range)
            if np.random.uniform() < self.keep_asp_ratio_prob:
                crop_asp_ratio = crop_img.shape[1] / crop_img.shape[0]  # w / h
                crop_width = crop_height * crop_asp_ratio
                if self.wratio_allowance[0] > crop_width:
                    crop_width = self.wratio_allowance[0]
                    crop_height = crop_width / crop_asp_ratio
                elif self.wratio_allowance[1] < crop_width:
                    crop_width = self.wratio_allowance[1]
                    crop_height = crop_width / crop_asp_ratio
            else:
                crop_width = np.random.uniform(*self.wratio_allowance)
            crop_size_wh: Tuple[int, int] = (int(crop_width * iw), int(crop_height * ih))  # (width, height), abs
            crop_rgb, crop_mask = crop_img[:, :, :3], crop_img[:, :, 3:]
            crop_rgb = mmcv.imresize(img=crop_rgb, size=crop_size_wh, interpolation='bicubic')
            crop_mask = mmcv.imresize(img=crop_mask, size=crop_size_wh, interpolation='nearest')
            crop_center_x, crop_center_y = self._sample_2d(crop_width / 2, 1 - crop_width / 2, 0,
                                                           1 - crop_height / 2, paste_num=i, total_paste=num_pastes,
                                                           h_horizon=max_sky_y)
            crop_left_top_x = int((crop_center_x * iw) - (crop_size_wh[0] / 2))
            crop_left_top_y = int((crop_center_y * ih) - (crop_size_wh[1] / 2))
            if crop_left_top_x < 0:
                crop_offset_x = -crop_left_top_x
                crop_rgb, crop_mask = crop_rgb[:, crop_offset_x:], crop_mask[:, crop_offset_x:]
                crop_left_top_x = 0
            if crop_left_top_y < 0:
                crop_offset_y = -crop_left_top_y
                crop_rgb, crop_mask = crop_rgb[crop_offset_y:, :], crop_mask[crop_offset_y:, :]
                crop_left_top_y = 0
            crop_right_bottom_x, crop_right_bottom_y = crop_left_top_x + crop_rgb.shape[1], crop_left_top_y + \
                                                       crop_rgb.shape[0]
            final_mask = np.bitwise_and(crop_mask, np.isin(
                sem[crop_left_top_y:crop_right_bottom_y, crop_left_top_x:crop_right_bottom_x],
                self.over_layer_indices))
            sem[crop_left_top_y:crop_right_bottom_y, crop_left_top_x:crop_right_bottom_x] = np.where(final_mask,
                                                                                                     self.crop_label,
                                                                                                     sem[
                                                                                                     crop_left_top_y:crop_right_bottom_y,
                                                                                                     crop_left_top_x:crop_right_bottom_x])
            img[crop_left_top_y:crop_right_bottom_y, crop_left_top_x:crop_right_bottom_x] = np.where(
                final_mask[:, :, np.newaxis], crop_rgb,
                img[crop_left_top_y:crop_right_bottom_y, crop_left_top_x:crop_right_bottom_x])

        results["img"], results["gt_semantic_seg"] = img, sem
        return results


if __name__ == "__main__":
    # For debug purposes
    pass
