"""
Evaluate multiple datasets. Get the distributions of the metrics.
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
import numpy as np
from tqdm import tqdm

from mmseg.apis.inference import init_segmentor, inference_segmentor
from uda_helpers.evaluate import MetricWithMainObject, StatisticsWithMainObject, evaluate_main_object
from uda_helpers.io import read_yaml, get_anno
from uda_helpers.visualize import save_results, initialize_figures, plot_distribution
from uda_helpers.utils import BBox, expand_bbox, get_largest_connected_component_in_center

from .evaluate import update_legacy_cfg

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

    parser.add_argument_group("Dataset")
    parser.add_argument("--ds_base_dir", type=str, default="", help="base directory for the datasets")
    parser.add_argument("--datasets", type=str, nargs='+',
                        default=["andreas-tree1", "andreas-tree2", "andreas-tree3", "andreas-tree4"],
                        help="datasets to be evaluated. Should be located at ds_base_dir")

    parser.add_argument("--exclusion", type=str, nargs='+', default=['../daformer/tools/exclusion.txt'],
                        help="images to be excluded from evaluation")

    parser.add_argument_group("Evaluation")
    parser.add_argument("--center_class", type=str, default="trunk", help="center class for the crop")
    parser.add_argument("--expand_ratio_x", type=float, default=0.3,
                        help="expand ratio for x-axis. 0.5 -> 50% expansion")
    parser.add_argument("--expand_ratio_y", type=float, default=0.2,
                        help="expand ratio for y-axis. 0.5 -> 50% expansion")

    parser.add_argument_group("Output")
    parser.add_argument("--output",
                        nargs='+',
                        type=str,
                        choices=["npy", "conf", "img", "comp_img", "none"],
                        default=["none"],
                        help="output to be saved")
    parser.add_argument(
        '--out_dir', default='out', help='directory where painted images will be saved')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.8,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument("--plot_distribution", action="store_true",
                        help="Plot the distribution of the metrics (e.g., mIoU)")

    parser.add_argument_group("Misc")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    if args.exp != '':
        config = os.path.join('work_dirs', 'local-basic', args.exp, f'{args.exp}.json')
        checkpoint = os.path.join('work_dirs', 'local-basic', args.exp, 'latest.pth' if args.it == -1 else f'iter_{args.it}.pth')
    elif args.config != '' and args.checkpoint != '':
        config = args.config
        checkpoint = args.checkpoint
    else:
        raise ValueError("Either exp or config and checkpoint has to be defined.")

    model = init_segmentor(config, checkpoint,
                           device=args.device,
                           config_updater=update_legacy_cfg,
                           revise_checkpoint=[(r'^module\.', ''), ('model.', '')])

    global classes, palettes
    del palettes['unlabeled']
    del classes['unlabeled']

    outputs = None
    if 'none' in args.output and len(args.output) != 1:
        outputs = None
        print(f"Warning: 'none' is in outputs. Ignoring other outputs.")
    else:
        outputs = args.output

    show_dirs, show_dir = None, None
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

    ds_base_dir = pathlib.Path(args.ds_base_dir)
    datasets = []
    for d in args.datasets:
        if not (ds_base_dir / d).exists():
            raise ValueError(f"Dataset {d} not found at {ds_base_dir}")
        datasets.append(ds_base_dir / d)

    img_exclusions = []
    if len(args.exclusion) > 0:
        for e in args.exclusion:
            if e.endswith('.txt'):
                with open(e, 'r') as f:
                    img_exclusions += [l.strip() for l in f.readlines()]
            else:
                raise NotImplementedError("Only txt file is supported for exclusion.")

    if args.plot_distribution:
        distributions = [{'mIoU': [], 'main_object_IoU': []} for _ in range(len(datasets))]

    if 'comp_img' in outputs:
        fig2, fig3 = initialize_figures()
    else:
        fig2, fig3 = None, None

    for j, ds in enumerate(datasets):
        images = [i for i in (ds / "images").rglob("*.[jJ][pP][gG]") if i.name not in img_exclusions]
        annotations = [i for i in (ds / "annotations").rglob("*.png")]

        stat = StatisticsWithMainObject(classes=classes)

        if outputs is not None and 'none' not in outputs:
            show_dir = show_dirs / ds.name
            show_dir.mkdir(exist_ok=True)
        for img in tqdm(images, total=len(images)):
            im = img
            img = cv2.imread(im.as_posix())
            results = inference_segmentor(model, img)
            logits = results[1][0]
            pred = results[0][0]
            anno = get_anno(im, annotations)
            if anno is None:
                raise ValueError(f"Annotation not found for {im.name}")
            target_mask = (anno == classes[args.center_class])
            main_object_mask, target_bbox = get_largest_connected_component_in_center(target_mask, img_shape=anno.shape[:2])
            expanded_bbox = expand_bbox(target_bbox, expand_ratio_x=args.expand_ratio_x,
                                        expand_ratio_y=args.expand_ratio_y, img_shape=anno.shape[:2])
            pred = pred[expanded_bbox.y:expanded_bbox.y + expanded_bbox.h, expanded_bbox.x:expanded_bbox.x + expanded_bbox.w]
            anno = anno[expanded_bbox.y:expanded_bbox.y + expanded_bbox.h, expanded_bbox.x:expanded_bbox.x + expanded_bbox.w]
            img = img[expanded_bbox.y:expanded_bbox.y + expanded_bbox.h, expanded_bbox.x:expanded_bbox.x + expanded_bbox.w]
            main_object_mask = main_object_mask[expanded_bbox.y:expanded_bbox.y + expanded_bbox.h, expanded_bbox.x:expanded_bbox.x + expanded_bbox.w]
            metric = evaluate_main_object(pred, anno, main_object_mask=main_object_mask, classes=classes,
                                          center_class=args.center_class,
                                          ignored_classes=['low-vegetation', 'sky', 'others', 'building'])
            stat.update(metric)
            save_results(outputs,
                         show_dirs=show_dir,
                         result=metric,
                         img=img,
                         img_name=im.name,
                         pred=pred,
                         conf=logits,
                         anno=anno,
                         palettes=palettes,
                         opacity=args.opacity,
                         fig2=fig2,
                         fig3=fig3,
                         )
            if args.plot_distribution:
                distributions[j]['mIoU'].append(metric.mean_iou)
                if isinstance(metric, MetricWithMainObject):
                    distributions[j]['main_object_IoU'].append(metric.main_object_iou)
        print("------------ " + ds.name + " ------------\n")
        print(stat.average)
        print("\n")
        print("------------")

    if args.plot_distribution:
        plot_distribution(distributions, save_path='./distribution.png', series=[d.name for d in datasets])



if __name__ == '__main__':
    main()
