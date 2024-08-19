"""
Same use case as daformer/tools/evaluate.py, but with mmseg's evaluation,
as this mmsegmentation package has different version (>1.0).
"""

import argparse
import os
import shutil
import pathlib
from typing import Dict

import cv2
from tqdm import tqdm

from mmseg.apis import init_model, inference_model
from uda_helpers.utils import add_parser_arguments
from uda_helpers.evaluate import init_stats, EvalProcessor
from uda_helpers.io import read_yaml, get_anno, print_eval_results

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

    add_parser_arguments(parser)

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

    images = [i for i in images if i.name not in img_exclusions]
    if args.exclude_air:
        images = [img for img in images if 'air' not in img.name]

    eval_processor = EvalProcessor(outputs=outputs, show_dir=show_dirs,
                                   classes=classes, palettes=palettes, depth_thresholds=args.depth_threshold)

    for img in tqdm(images, total=len(images)):
        im = img
        img = cv2.imread(im.as_posix())
        result = inference_model(model, im.as_posix())
        logits = result.seg_logits.data
        pred = logits.argmax(dim=0).squeeze().cpu().numpy()
        anno = get_anno(im, annotations)
        depth_map = get_anno(im, depth_maps) if depth_maps is not None else None

        if not args.with_main_object:
            eval_processor.process_whole_image(
                im=img,
                img_name=im.name,
                pred=pred,
                anno=anno,
                depth_map=depth_map,
                show_dir=show_dirs,
                conf=logits.squeeze().cpu().numpy(),
                stat=stats,
                ignored_classes=args.ignore_class,
                opacity=args.opacity,
                show_ori_img=args.show_ori_img,
            )
        else:
            eval_processor.process_crop(
                im=img,
                img_name=im.name,
                pred=pred,
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
