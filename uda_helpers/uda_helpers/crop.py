# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image


@dataclass
class Crop:
    img_name: str
    crop_id: int
    x: int
    y: int
    w: int
    h: int

    _pred_path: Optional[str] = None
    _conf_path: Optional[str] = None
    _pred: Optional[np.ndarray] = None
    _conf: Optional[np.ndarray] = None

    _metrics: Optional[dict] = None

    @classmethod
    def from_txt_line(cls, line: str) -> "Crop":
        img_name, x, y, w, h = line.split(',')
        img_name, crop_id = img_name[:img_name.rfind('_')], img_name[img_name.rfind('_') + 1:]
        return cls(img_name, int(crop_id), int(x), int(y), int(w), int(h))

    @property
    def full_name(self) -> str:
        return self.img_name + '_' + str(self.crop_id)

    @property
    def pred(self) -> np.ndarray:
        return self._pred

    @property
    def conf(self) -> np.ndarray:
        return self._conf

    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    @property
    def rel_center(self) -> Tuple[int, int]:
        # relative center
        return self.w // 2, self.h // 2

    def set_metrics(self, metrics: Dict[str, float]):
        self._metrics = metrics

    def get_metrics(self, **kwargs):
        pass

    def set_prediction_path(self, pred: str, conf: Optional[str] = None):
        self._pred_path = pred
        if conf is not None:
            self._conf_path = conf

    def convert_to_half_size(self):
        self.x //= 2
        self.y //= 2
        self.w //= 2
        self.h //= 2
        if self._pred_path is not None:
            pred = np.load(self._pred_path)
            self._pred = cv2.resize(pred, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if self._conf_path is not None:
            conf = np.load(self._conf_path)
            conf = torch.from_numpy(conf)
            conf = TF.resize(conf, [self.h, self.w], interpolation=Image.NEAREST)
            self._conf = conf.numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
