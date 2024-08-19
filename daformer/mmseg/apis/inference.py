# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Override palette, classes, and state dict keys
import pathlib


import matplotlib.pyplot as plt
import numpy as np
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor


def init_segmentor(config,
                   checkpoint=None,
                   device='cuda:0',
                   classes=None,
                   palette=None,
                   revise_checkpoint=[(r'^module\.', '')],
                   config_updater=None):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    if config_updater is not None:
        config = config_updater(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(
            model,
            checkpoint,
            map_location='cpu',
            revise_keys=revise_checkpoint)
        model.CLASSES = checkpoint['meta']['CLASSES'] if classes is None \
            else classes
        model.PALETTE = checkpoint['meta']['PALETTE'] if palette is None \
            else palette
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def _single_image_inference(model, img, device, pipeline):
    # prepare data
    data = dict(img=img)
    data = pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def inference_segmentor(model, img, print_progress=False):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
        print_progress (bool): Whether to print inference progress.

    Returns:
        dict: The segmentation result. Key: filename, value: List[Tensor]
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    if isinstance(img, str) or isinstance(img, np.ndarray) or isinstance(img, pathlib.Path):
        return _single_image_inference(model, img, device, test_pipeline)
    else:
        results = {}
        for i in img:
            if print_progress:
                print(f"Inference on {i}")
            if isinstance(i, pathlib.Path):
                i = i.as_posix()
            elif isinstance(i, np.ndarray):
                raise RuntimeError("Unaccepted type of inference image as it is not hashable as a dict key. "
                                   "Accepted types: str, pathlib.Path")
            results[i] = _single_image_inference(model, i, device, test_pipeline)
        return results


def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)
