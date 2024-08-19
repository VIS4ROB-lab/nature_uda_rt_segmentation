"""
Calculate the metrics around crops instead of the entire image.
"""

import argparse
import os
import shutil
import pathlib
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

from mmseg.apis import init_model, inference_model
from uda_helpers.evaluate import MetricWithMainObject, StatisticsWithMainObject, evaluate_main_object
from uda_helpers.io import read_yaml, get_anno
from uda_helpers.visualize import save_results, initialize_figures
from uda_helpers.utils import BBox, expand_bbox, get_largest_connected_component_in_center

classes_yaml = '../dataset/classes.yaml'

yaml_content = read_yaml(classes_yaml, classes=True, palettes=True, pop_unlabeled=False)
classes, palettes = yaml_content['classes'], yaml_content['palettes']


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation. See the docstring for more details.')

    parser.add_argument_group("Basic")
    parser.add_argument("-c", "--config", type=str, default='',
                        help="model config file path", required=True)
    parser.add_argument("-w", "--checkpoint", type=str, default='',
                        help="path to the model weights/checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0", help="device used for inference")

    parser.add_argument_group("Dataset")
    parser.add_argument("--image_dir", type=str, default="./data/Apple_Farm_Real_psl/images/validation")
    parser.add_argument("--img_extension", type=str, default="JPG", help="image extension for dataset")
    parser.add_argument("--anno_dir", type=str, default="./data/Apple_Farm_Real_psl/annotations/validation")

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
                        help="Plot the distribution of the mIoU, mAcc, canopy coverage and adjusted trunk accuracy.")

    parser.add_argument_group("Misc")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    model = init_model(config=args.config, checkpoint=args.checkpoint, device=args.device)

    global classes, palettes
    del palettes['unlabeled']
    del classes['unlabeled']

    outputs = None
    if 'none' in args.output and len(args.output) != 1:
        outputs = None
        print(f"Warning: 'none' is in outputs. Ignoring other outputs.")
    else:
        outputs = args.output

    show_dirs = None
    if outputs is not None and 'none' not in outputs:
        show_dirs = pathlib.Path(args.out_dir) if args.out_dir.startswith('/') else pathlib.Path(
            os.path.join('work_dirs', args.out_dir))
        if show_dirs.exists():
            for item in show_dirs.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        show_dirs.mkdir(exist_ok=True, parents=True)

    images = list(pathlib.Path(args.image_dir).rglob(f"*.{args.img_extension}"))
    annotations = list(pathlib.Path(args.anno_dir).rglob("*.png"))

    img_exclusions = []
    if len(args.exclusion) > 0:
        for e in args.exclusion:
            if e.endswith('.txt'):
                with open(e, 'r') as f:
                    img_exclusions += [l.strip() for l in f.readlines()]
            else:
                raise NotImplementedError("Only txt file is supported for exclusion.")

    images = [i for i in images if i.name not in img_exclusions]

    # if args.plot_distribution:
    #     distributions = {'mIoU': [], 'mAcc': [], 'cc': [], 'ata': []}

    if 'comp_img' in outputs:
        fig2, fig3 = initialize_figures()
    else:
        fig2, fig3 = None, None

    stat = StatisticsWithMainObject(classes=classes)

    for img in tqdm(images, total=len(images)):
        im = img
        img = cv2.imread(im.as_posix())
        result = inference_model(model, im.as_posix())
        logits = result.seg_logits.data
        pred = logits.argmax(dim=0).squeeze().cpu().numpy()
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
                     show_dirs=show_dirs,
                     result=metric,
                     img=img,
                     img_name=im.name,
                     pred=pred,
                     conf=logits.squeeze().cpu().numpy(),
                     anno=anno,
                     palettes=palettes,
                     opacity=args.opacity,
                     fig2=fig2,
                     fig3=fig3,
                     )
    print(stat.average)
    print("\n")


if __name__ == '__main__':
    main()
