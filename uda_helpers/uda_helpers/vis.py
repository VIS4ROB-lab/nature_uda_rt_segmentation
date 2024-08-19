"""
Direct call for visualizing semantic maps with command line.
"""

# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import argparse
import pathlib
from functools import partial

import cv2
import numpy as np

from .io import read_yaml
from .visualize import visualize_sem_with_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the semantic maps.")
    parser.add_argument("--yaml", type=str, required=True, help="Path to the yaml file, defining the palettes.")
    parser.add_argument("--img", type=str, required=True, help="Path to the original image or directory.")
    parser.add_argument("--sem", type=str, required=True, help="Path to the semantic segmentation results or directory.")
    parser.add_argument("--out", type=str, default='./out', help="Path to the output directory.")
    parser.add_argument("--opacity", type=float, default=0.5, help="Opacity of the overlaid segmentation results.")

    args = parser.parse_args()

    palettes = read_yaml(args.yaml, classes=False, palettes=True)['palettes']

    img, sem = pathlib.Path(args.img), pathlib.Path(args.sem)
    img = list(img.rglob("*.[jJ][pP][gG]")) if img.is_dir() else [img]
    sem = list(sem.rglob("*.png")) + list(sem.rglob("*.npy")) if sem.is_dir() else [sem]

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    if args.opacity <= 0 or args.opacity > 1:
        raise ValueError("Invalid opacity value! Expecting a value in (0, 1] range.")

    read_png = partial(cv2.imread, flags=cv2.IMREAD_UNCHANGED)
    load_sem = {'.png': read_png, '.npy': np.load}

    if len(sem) == len(img) == 1:
        im = cv2.imread(img[0].as_posix(), -1)
        se = load_sem[sem[0].suffix](sem[0].as_posix())
        overlaid = visualize_sem_with_img(img=im, sem_seg=se, opacity=args.opacity, palette=palettes)
        cv2.imwrite((out_dir / (sem[0].stem + ".png")).as_posix(), overlaid)
    else:
        for s in sem:
            img_path = list(filter(lambda x: x.stem == s.stem, img))
            if len(img_path) == 0:
                print(f"Image not found for {s.name}!")
                continue
            img_path = img_path[0]
            se = load_sem[s.suffix](s.as_posix())
            im = cv2.imread(img_path.as_posix(), -1)
            overlaid = visualize_sem_with_img(img=im, sem_seg=se, opacity=args.opacity, palette=palettes)
            cv2.imwrite((out_dir / (s.stem + ".png")).as_posix(), overlaid)