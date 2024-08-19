"""
New infer script for mmsegmentation.
"""

import argparse
import os
import pathlib
import shutil

import numpy as np
import cv2
from mmseg.apis import init_model, inference_model

from uda_helpers.timing import timing, time_statistics
from uda_helpers.io import read_yaml
from uda_helpers.visualize import save_results


classes_yaml = '../dataset/classes.yaml'

yaml_content = read_yaml(classes_yaml, classes=True, palettes=True, pop_unlabeled=False)
classes, palettes = yaml_content['classes'], yaml_content['palettes']


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for mmsegmentation")

    parser.add_argument_group()
    parser.add_argument_group("Basic")
    parser.add_argument("-c", "--config", type=str, default='',
                        help="model config file path", required=True)
    parser.add_argument("-w", "--checkpoint", type=str, default='',
                        help="path to the model weights/checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0", help="device used for inference")

    parser.add_argument_group("Dataset")
    parser.add_argument("--image_dir", type=str, default="./data/Apple_Farm_Real_psl/images/validation")

    parser.add_argument_group("Output")
    parser.add_argument("--output",
                        nargs='+',
                        type=str,
                        choices=["npy", "conf", "img", "png", "none"],
                        default=["none"],
                        help="output to be saved")
    parser.add_argument(
        '--out_dir', default='out', help='directory where painted images will be saved')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.8,
        help='Opacity of painted segmentation map. In (0, 1] range.')

    parser.add_argument("--timing", action="store_true", help="perform a timing test")

    return parser.parse_args()


@timing
def _timed_inference(model, x: np.ndarray) -> np.ndarray:
    return inference_model(model, x)


def _main():
    args = _parse_args()
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
        show_dirs = pathlib.Path(args.out_dir) if pathlib.Path(args.out_dir).is_dir() \
            else pathlib.Path(os.path.join('work_dirs', args.out_dir))
        if show_dirs.exists():
            for item in show_dirs.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        show_dirs.mkdir(exist_ok=True, parents=True)

    images = list(pathlib.Path(args.image_dir).rglob(f"*.JPG")) + list(pathlib.Path(args.image_dir).rglob(f"*.jpg"))

    if args.timing:
        print("---------- Performing timing test ----------")
        for image in images:
            img = cv2.imread(image.as_posix(), -1)
            _ = _timed_inference(model, img)
        print(time_statistics(_timed_inference.times))
    else:
        for image in images:
            img = cv2.imread(image.as_posix(), -1)
            result = inference_model(model, img)
            logits = result.seg_logits.data
            pred = logits.argmax(dim=0).squeeze().cpu().numpy()
            save_results(outputs,
                         show_dirs=show_dirs,
                         result=None,
                         img=img,
                         img_name=image.name,
                         pred=pred,
                         conf=logits,
                         anno=None,
                         palettes=palettes,
                         opacity=args.opacity,
                         )


if __name__ == "__main__":
    _main()
