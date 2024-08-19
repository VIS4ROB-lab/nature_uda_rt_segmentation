import argparse
import os
import pathlib
import re


import numpy as np
import cv2
from mmseg.apis import MMSegInferencer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='./out/segformer_mit-b0_plant_seg.py',
                        help="model or model config file path")
    parser.add_argument("-w", "--weights", type=str, default='./out/iter_1000.pth',
                        help="path to the model weights", required=False)
    parser.add_argument("-p", "--img_path", type=str, default='./data/plant_seg/images/',
                        help="base path for images")
    parser.add_argument("-i", "--image", type=str, nargs='+', default='validation/000001.png',
                        help="path to the image")
    parser.add_argument("--resize", type=float, default=1, help="factor to resize the input images")
    parser.add_argument("-b", "--max_batch", type=int, default=16, help="max batch size of inference")
    parser.add_argument("--mask", action="store_true",
                        help='enable masking mode to make masks using inferences for a specific class')
    parser.add_argument("--mask_cls", type=int, default=2,
                        help="class mask to make. only valid in masking mode. sky is label 2 in ADE20K")

    args = parser.parse_args()
    if not args.mask:
        classes = ("sky", "terrain", "low-vegetation", "trunk", "canopy", "building", "others")
        palette = ([60, 112, 164], [152, 251, 152], [147, 217, 163], [194, 118, 100], [97, 135, 110], [244, 242, 222], [250, 170, 30])
        inferencer = MMSegInferencer(model=args.model,
                                     weights=args.weights,
                                     )
        out_path = './out/inference/vis'
    else:
        inferencer = MMSegInferencer(model=args.model,
                                     weights=args.weights,)
        out_path = './out/inference/mask'
    pathlib.Path(out_path).mkdir(exist_ok=True, parents=True)
    images = []
    if args.image and args.image[0] != 'x':
        images = [os.path.join(args.img_path, img) for img in args.image] if args.img_path else args.image
    elif args.image[0] == 'x' and args.img_path:
        print(f"Walking through {args.img_path}")
        images = [os.path.join(args.img_path, im) for im in os.listdir(args.img_path) if re.match(r"^.*\.(jpg|JPG|png|PNG)$", im)]

    total, current = len(images), 0
    print(f"Inferring {len(images)} images")
    while len(images) > 0:
        batch, images = images[:args.max_batch], images[args.max_batch:]
        im_batch = [cv2.imread(img_path, -1) for img_path in batch]
        im_batch = [cv2.resize(img, (int(img.shape[1]/args.resize), int(img.shape[0]/args.resize)),
                               interpolation=cv2.INTER_LANCZOS4) for img in im_batch]
        im_batch = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in im_batch]
        current += len(im_batch)

        if not args.mask:
            result = inferencer(im_batch, show=False, opacity=0.8,
                                out_dir='./out/inference', img_out_dir='vis', pred_out_dir='pred')
        else:
            result = inferencer(im_batch, show=False)
            prediction = result['predictions']
            if len(batch) == 1:
                mask = np.where(prediction == args.mask_cls, 255, 0)
                cv2.imwrite(os.path.join(out_path, os.path.basename(batch[0])[:-3] + "png"), mask.astype(np.uint8))
                # np.save(os.path.join(out_path, os.path.basename(batch[0])), mask.astype(bool))
            else:
                raise NotImplementedError("Batching for this mode is not implemented")

        if current % 100 == 0:
            print(f"{current}/{total} finished")
