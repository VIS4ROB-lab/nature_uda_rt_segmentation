"""
Inference script for DAFormer models. Modified based on test.py
"""

# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import argparse
import os
import shutil
import pathlib
from typing import Union, List, Dict

import yaml
import numpy as np
import mmcv
from mmseg.apis.inference import init_segmentor, inference_segmentor

from uda_helpers.timing import timing, time_statistics


classes_yaml = '../dataset/classes.yaml'


def _load_yaml() -> dict:
    with open(classes_yaml, 'r') as f:
        config = yaml.safe_load(f)
    return config['palettes']


def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


@timing
def single_inference(model, img: np.ndarray) -> np.ndarray:
    return inference_segmentor(model, img)


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument("--device", default="cuda:0", help="device used for inference")
    # parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--out_dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        "--image",
        nargs='+',
        type=str,
        help="images or directory to be inferred"
    )
    parser.add_argument("--save_npy",
                        action="store_true",
                        help="Save npy files instead of images")
    parser.add_argument("--save_conf",
                        action="store_true",
                        help="Save confidence for the predicted classes for each pixel")
    parser.add_argument('--max_images', type=int, default=-1,
                        help='max number of images to inference. -1 for all')
    parser.add_argument("--max_queue", type=int, default=10,
                        help="max number of images in RAM in queue")

    parser.add_argument("--sky_only", action='store_true', help="Save only the sky mask")

    parser.add_argument("--raw_png", action='store_true', help="Save the original png")

    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument("--timing", action='store_true', help="Perform timing test. Ignore all outputs")

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def handle_results(img: Union[str, pathlib.Path],
                   pred: np.ndarray,
                   # conf: np.ndarray,
                   palette: Dict[str, List[int]],
                   opacity: float = 0.5) -> np.ndarray:
    img = mmcv.imread(img)
    palette = np.array(list(palette.values()))
    color_seg = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[pred == label, :] = color
    color_seg = color_seg[..., ::-1]  # RGB -> BGR
    return img * (1 - opacity) + color_seg * opacity


def queue_spliter(images: List[pathlib.Path], max_queue: int = 200) -> List[List[pathlib.Path]]:
    images_copy = images.copy()
    return [images_copy[i:i + max_queue] for i in range(0, len(images_copy), max_queue)]


def main():
    args = parse_args()
    palette = _load_yaml()
    del palette['unlabeled']
    model = init_segmentor(args.config, args.checkpoint,
                           device=args.device,
                           config_updater=update_legacy_cfg,
                           revise_checkpoint=[(r'^module\.', ''), ('model.', '')])

    images = args.image
    if len(images) == 1 and (inf_dir := pathlib.Path(images[0])).is_dir():
        images = list(inf_dir.rglob("*.[jJ][pP][gG]"))
    else:
        images = [pathlib.Path(i) for i in images]

    if args.timing:
        print("---------- Performing timing test ----------")
        for image in images:
            img = mmcv.imread(image)
            single_inference(model, img)
        print(time_statistics(single_inference.times))
    else:
        show_dirs = pathlib.Path(args.out_dir)
        if show_dirs.exists():
            for item in show_dirs.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        show_dirs.mkdir(exist_ok=True, parents=True)

        if args.max_images > 0:
            images = images[:args.max_images]
        for image_queue in queue_spliter(images, args.max_queue):
            results = inference_segmentor(model, image_queue)
            if isinstance(results, tuple) and len(results) == 2:
                raise NotImplementedError
            else:
                if not args.save_npy and not args.sky_only and not args.raw_png:
                    for img, out in results.items():
                        # out: tuple of 2, ([pred], [conf])
                        overlaid_img = handle_results(img, out[0][0], palette, opacity=args.opacity)
                        save_path = show_dirs / img.name if isinstance(img, pathlib.Path) else show_dirs / pathlib.Path(img).name
                        mmcv.imwrite(overlaid_img, save_path.as_posix())
                elif args.sky_only:
                    for img, out in results.items():
                        sky = np.where(out[0][0] == 0, 255, 0).astype(np.uint8)
                        save_path = show_dirs / (img.stem + '.png') if isinstance(img, pathlib.Path) else show_dirs / (pathlib.Path(img).stem + '.png')
                        mmcv.imwrite(sky, save_path.as_posix())
                elif args.raw_png:
                    for img, out in results.items():
                        save_path = show_dirs / (img.stem + '.png') if isinstance(img, pathlib.Path) else show_dirs / (pathlib.Path(img).stem + '.png')
                        mmcv.imwrite(out[0][0].astype(np.uint8), save_path.as_posix())
                else:
                    for img, out in results.items():
                        save_path = show_dirs / (img.stem + '.npy') if isinstance(img, pathlib.Path) else show_dirs / (pathlib.Path(img).stem + '.npy')
                        np.save(save_path.as_posix(), out[0][0].astype(np.uint8))
                        if args.save_conf:
                            conf_path = show_dirs / (img.stem + '_conf.npy') if isinstance(img, pathlib.Path) else show_dirs / (pathlib.Path(img).stem + '_conf.npy')
                            np.save(conf_path.as_posix(), out[1][0].astype(np.half))


if __name__ == '__main__':
    main()
