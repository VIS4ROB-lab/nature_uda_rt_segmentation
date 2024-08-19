"""
Evaluate pngs.
"""

# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import argparse
import os
import shutil
import pathlib

import cv2
from tqdm import tqdm

from .utils import add_parser_arguments
from .evaluate import init_stats, EvalProcessor
from .io import read_yaml, get_anno


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation.')

    parser.add_argument("--yaml", type=str, default='./dataset/classes.yaml',
                        help="path to the classes.yaml file")

    parser.add_argument_group("Basic")
    parser.add_argument("--pred_dir", type=str, nargs="+", help="prediction directory. should be png files")

    parser.add_argument_group("Dataset")
    parser.add_argument("--ds_base_dir", type=str, help="dataset base directory")
    parser.add_argument("--ds_dir", type=str, nargs="+", help="image directory(ies) for dataset")
    parser.add_argument("--ds_type", type=str, choices=["mmseg", "simple"], default="mmseg",
                        help="type of dataset. mmseg for mmseg datasets, simple for simple datasets")
    parser.add_argument("--depth_map_dir", type=str, nargs="+", help="depth map directory(ies) for dataset")
    parser.add_argument("--exclusion", type=str, nargs='+', default=["./daformer/tools/exclusion.txt"],
                        help="images to be excluded from evaluation")
    parser.add_argument("--exclude_air", action="store_true", help="exclude aerial images from evaluation")

    add_parser_arguments(parser, dataset=False)

    args = parser.parse_args()
    return args


def _main(args: argparse.Namespace) -> None:
    yaml_content = read_yaml(args.yaml, classes=True, palettes=True, pop_unlabeled=False)
    classes, palettes = yaml_content['classes'], yaml_content['palettes']
    del palettes['unlabeled']
    del classes['unlabeled']

    ds_base_dir = args.ds_base_dir
    ds_dirs = args.ds_dir
    images, annos, depths = [], [], []
    for d in ds_dirs:
        if args.ds_type == "mmseg":
            images.append([p for p in pathlib.Path(ds_base_dir, d, "images", "validation").rglob("*") if p.is_file()])
            annos.append([p for p in pathlib.Path(ds_base_dir, d, "annotations", "validation").rglob("*") if p.is_file()])
        elif args.ds_type == "simple":
            images.append([p for p in pathlib.Path(ds_base_dir, d, "images").rglob("*") if p.is_file()])
            annos.append([p for p in pathlib.Path(ds_base_dir, d, "annotations").rglob("*") if p.is_file()])
    if args.depth_map_dir is not None and len(args.depth_map_dir) > 0:
        depths = [p for d in args.depth_map_dir
                  for p in pathlib.Path(ds_base_dir, d).rglob("*.png") if p.is_file()]
    depth_thresholds = args.depth_threshold if args.depth_threshold is not None else [255]

    img_exclusions = []
    if len(args.exclusion) > 0:
        for e in args.exclusion:
            if e.endswith('.txt'):
                with open(e, 'r') as f:
                    img_exclusions += [l.strip() for l in f.readlines()]
            else:
                raise NotImplementedError("Only txt file is supported for exclusion.")

    pred_dir = [p for pth in args.pred_dir for p in pathlib.Path(pth).rglob("*.png")]

    outputs = None
    if 'none' in args.output and len(args.output) != 1:
        outputs = None
        print(f"Warning: 'none' is in outputs. Ignoring other outputs.")
    else:
        outputs = args.output

    show_dirs, show_dir = None, None
    if outputs is not None and 'none' not in outputs:
        show_dirs = pathlib.Path(args.out_dir) if len(args.out_dir) else pathlib.Path('./out')
        if show_dirs.exists():
            for item in show_dirs.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        show_dirs.mkdir(exist_ok=True, parents=True)

    for j, (imgs, annos) in enumerate(zip(images, annos)):
        stat = init_stats(classes=classes,
                          stat_groups=args.stat_groups,
                          depth_thresholds=depth_thresholds,
                          with_main_object=args.with_main_object)
        if outputs is not None and 'none' not in outputs:
            show_dir = show_dirs / ds_dirs[j]
            show_dir.mkdir(exist_ok=True)

        eval_processor = EvalProcessor(outputs=outputs, show_dir=show_dir,
                                       classes=classes, palettes=palettes, depth_thresholds=args.depth_threshold,)

        for img in tqdm(imgs, total=len(imgs)):
            if img.name in img_exclusions:
                continue
            if "air" in img.name and args.exclude_air:
                continue
            im = cv2.imread(img.as_posix(), -1)
            anno = get_anno(img, annos)
            if anno is None:
                raise ValueError(f"Annotation not found for {img}")
            depth_map = get_anno(img, depths) if len(depths) > 0 else None

            results = get_anno(img, pred_dir)
            if results is None:
                raise ValueError(f"Prediction not found for {img}")
            pred = results
            conf = None

            im = cv2.resize(im, (2250, 1500), interpolation=cv2.INTER_LANCZOS4)
            pred = cv2.resize(pred, (2250, 1500), interpolation=cv2.INTER_NEAREST)
            anno = cv2.resize(anno, (2250, 1500), interpolation=cv2.INTER_NEAREST)
            depth_map = cv2.resize(depth_map, (2250, 1500), interpolation=cv2.INTER_NEAREST) if depth_map is not None else None

            if not args.with_main_object:
                eval_processor.process_whole_image(
                    im=im,
                    img_name=img.name,
                    pred=pred,
                    anno=anno,
                    conf=conf,
                    stat=stat,
                    show_dir=show_dir,
                    depth_map=depth_map,
                    ignored_classes=args.ignore_class,
                    opacity=args.opacity,
                    show_ori_img=args.show_ori_img,
                )
            else:
                eval_processor.process_crop(
                    im=im,
                    img_name=img.name,
                    pred=pred,
                    anno=anno,
                    center_class=args.center_class,
                    expand_ratio_x=args.expand_ratio_x,
                    expand_ratio_y=args.expand_ratio_y,
                    stat=stat,
                    ignored_classes=args.ignore_class,
                    opacity=args.opacity,
                    show_ori_img=args.show_ori_img,
                )
        print("------------ " + ds_dirs[j] + " ------------\n")
        if not args.with_main_object:
            for d, s in stat.items():
                for g, _s in s.items():
                    if _s.get_count() == 0:
                        continue
                    print(f"Group: {g}, Depth: {d}")
                    print(_s.average)
                    print("----")
                print("------------")
            print("\n")
        else:
            for g in stat.keys():
                if stat[g].get_count() == 0:
                    continue
                print(f"Group: {g}")
                print(stat[g].average)
                print("----")


if __name__ == '__main__':
    _main(parse_args())
