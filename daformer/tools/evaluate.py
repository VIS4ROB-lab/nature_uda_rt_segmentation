"""
An alternative to the official evaluation script.

Instead of using the official test script,
the following script is used to evaluate the model to unify the evaluation process for all steps.

You may run this from the daformer directory with the following command:
python3 -m tools.evaluate --exp {YOUR_EXPERIMENT_NAME}

You may read the arguments in the parse_args function for more details.
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

from mmseg.apis.inference import init_segmentor, inference_segmentor
from uda_helpers.evaluate import init_stats, EvalProcessor
from uda_helpers.io import read_yaml, get_anno, print_eval_results
from uda_helpers.utils import update_legacy_cfg, add_parser_arguments
from .helpers import fetch_experiment

classes_yaml = '../dataset/classes.yaml'

yaml_content = read_yaml(classes_yaml, classes=True, palettes=True, pop_unlabeled=False)
classes, palettes = yaml_content['classes'], yaml_content['palettes']


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation. See the docstring for more details.')

    parser.add_argument_group("Basic")
    parser.add_argument("--exp", type=str, default='', help="Experiment name")
    parser.add_argument("--it", default=-1, type=int, help="Iteration number. -1 for latest")
    parser.add_argument('--config', default='', help='test config file path')
    parser.add_argument('--checkpoint', default='', help='checkpoint file')
    parser.add_argument("--device", default="cuda:0", help="device used for inference")

    add_parser_arguments(parser)

    parser.add_argument_group("Misc")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    if args.exp != '':
        if not os.path.exists(os.path.join('work_dirs', 'local-basic', args.exp)):
            fetch_experiment(args.exp)
        config = os.path.join('work_dirs', 'local-basic', args.exp, f'{args.exp}.json')
        checkpoint = os.path.join('work_dirs', 'local-basic', args.exp, 'latest.pth' if args.it == -1 else f'iter_{args.it}.pth')
    elif args.config != '' and args.checkpoint != '':
        config = args.config
        checkpoint = args.checkpoint
    else:
        raise ValueError("Either exp or config and checkpoint has to be defined.")

    global classes, palettes
    del palettes['unlabeled']
    del classes['unlabeled']
    model = init_segmentor(config, checkpoint,
                           device=args.device,
                           config_updater=update_legacy_cfg,
                           revise_checkpoint=[(r'^module\.', ''), ('model.', '')])

    outputs = None
    if 'none' in args.output and len(args.output) != 1:
        outputs = None
        print(f"Warning: 'none' is in outputs. Ignoring other outputs.")
    else:
        outputs = args.output
    show_dirs = None
    if outputs is not None and 'none' not in outputs:
        show_dirs = pathlib.Path(args.out_dir) if args.out_dir.startswith('/') else pathlib.Path(
            os.path.join('work_dirs', 'local-basic', args.exp, args.out_dir))
        if show_dirs.exists():
            for item in show_dirs.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        show_dirs.mkdir(exist_ok=True, parents=True)

    images = list(pathlib.Path(args.image_dir).rglob(f"*.{args.img_extension}"))
    annotations = list(pathlib.Path(args.anno_dir).rglob("*.png"))
    if len(args.depth_map_dir) and (depth_map_dir := pathlib.Path(args.depth_map_dir)).exists():
        depth_maps = list(depth_map_dir.rglob("*.png"))
    else:
        depth_maps = None

    depth_thresholds = args.depth_threshold if args.depth_threshold is not None else [255]
    stats = init_stats(classes=classes, stat_groups=args.stat_groups, depth_thresholds=depth_thresholds,
                       with_main_object=args.with_main_object)

    img_exclusions = []
    if len(args.exclusion) > 0:
        for e in args.exclusion:
            if e.endswith('.txt'):
                with open(e, 'r') as f:
                    img_exclusions += [l.strip() for l in f.readlines()]
            else:
                raise NotImplementedError("Only txt file is supported for exclusion.")

    images = [img for img in images if img.name not in img_exclusions]
    if args.exclude_air:
        images = [img for img in images if 'air' not in img.name]

    eval_processor = EvalProcessor(outputs=outputs, show_dir=show_dirs,
                                   classes=classes, palettes=palettes, depth_thresholds=args.depth_threshold)

    for im in tqdm(images, total=len(images)):
        img = cv2.imread(im.as_posix())
        results = inference_segmentor(model, img)
        anno = get_anno(im, annotations)
        if anno is None:
            raise ValueError(f"Annotation not found for {im.name}")
        depth_map = get_anno(im, depth_maps) if depth_maps is not None else None

        if not args.with_main_object:
            eval_processor.process_whole_image(
                im=img,
                img_name=im.name,
                pred=results[0][0],
                anno=anno,
                depth_map=depth_map,
                show_dir=show_dirs,
                conf=results[1][0],
                stat=stats,
                ignored_classes=args.ignore_class,
                opacity=args.opacity,
                show_ori_img=args.show_ori_img,
            )
        else:
            eval_processor.process_crop(
                im=img,
                img_name=im.name,
                pred=results[0][0],
                anno=anno,
                center_class=args.center_class,
                expand_ratio_x=args.expand_ratio_x,
                expand_ratio_y=args.expand_ratio_y,
                stat=stats,
                ignored_classes=args.ignore_class,
                opacity=args.opacity,
                show_ori_img=args.show_ori_img,
            )
    print_eval_results(stats, with_main_object=args.with_main_object)


if __name__ == '__main__':
    main()
