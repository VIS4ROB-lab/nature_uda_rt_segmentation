# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import pathlib
import numpy as np
import cv2


workdir = 'data/apple_farm_summer_clean'
image_suffix = '.JPG'
anno_suffix = '.png'

workdir = pathlib.Path(workdir)
workdir_img, workdir_anno = workdir / "images", workdir / "annotations"
workdir_img_train, workdir_anno_train = workdir_img / "training", workdir_anno / "training"
workdir_img_val, workdir_anno_val = workdir_img / "validation", workdir_anno / "validation"

for image_path, anno_path in ((workdir_img_train, workdir_anno_train), (workdir_img_val, workdir_anno_val)):
    print(f"Working on {image_path} -> {anno_path}")
    anno_path.mkdir(exist_ok=True, parents=True)
    for image in image_path.rglob("*" + image_suffix):
        img = cv2.imread(image.as_posix(), cv2.IMREAD_UNCHANGED)
        dummy_anno = np.zeros(img.shape[:2])
        cv2.imwrite((anno_path / (image.stem + anno_suffix)).as_posix(), dummy_anno)
