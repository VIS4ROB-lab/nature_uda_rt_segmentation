# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import List, Dict, Optional, Union
import pathlib
import yaml
import cv2
import numpy as np

from .types import Classes
from .evaluate import DictDepthStats, DictGroupMainObjStats

def read_yaml(
        yaml_path: str,
        classes: bool = True,
        palettes: bool = False,
        cls_values: bool = False,
        sem_map: str = '',
        pop_unlabeled: bool = True
) -> Dict:
    """
    Read "classes.yaml" and return the requested content as a list of dictionaries.

    Args:
        yaml_path (str): Path to the "classes.yaml" file.
        classes (bool): Whether to return the classes.
        palettes (bool): Whether to return the palettes.
        cls_values (bool): Whether to return the class values.
        sem_map (str): Name of the semantic mapping to .
        pop_unlabeled (bool): Whether to remove the "unlabeled" class and palette.

    Returns:
        Dict: Dict of the requested content.
    """
    with open(yaml_path, 'r') as f:
        yaml_content = yaml.safe_load(f)
    if 'classes' not in yaml_content:
        raise ValueError("Essential 'classes' not found in classes.yaml.")
    if 'palettes' not in yaml_content:
        raise ValueError("Essential 'palettes' not found in classes.yaml.")
    if pop_unlabeled:
        yaml_content['classes'].pop('unlabeled')
        yaml_content['palettes'].pop('unlabeled')
    results = {}
    _class_values = set(list(yaml_content['classes'].values()))
    if classes:
        results['classes'] = yaml_content['classes']
    if palettes:
        results['palettes'] = yaml_content['palettes']
    if cls_values:
        results['cls_values'] = _class_values
    if len(sem_map):
        try:
            sem_map = yaml_content['sem_map'][sem_map]
        except KeyError:
            raise ValueError(f"Semantic mapping {sem_map} does not exist in classes.yaml. Please check again.")
        if not set(sem_map.values()).issubset(_class_values):
            raise RuntimeError("Not all values of the semantic mapping are included in the classes! "
                               "Please check classes.yaml")
        results['sem_map'] = sem_map
    return results


def png2jpg(image_file: pathlib.Path, target_path:pathlib.Path) -> None:
    image = cv2.imread(image_file.as_posix(), -1)
    cv2.imwrite(target_path.as_posix(), image)


def get_anno(img: Union[str, pathlib.Path], annotations: List[pathlib.Path]) -> Optional[np.ndarray]:
    """
    Get the annotation corresponding to the image, given the list of annotations and the name of the image.

    Args:
        img (Union[str, pathlib.Path]): Path to the image.
        annotations (List[pathlib.Path]): List of annotations.
    """
    anno = list(filter(lambda x: x.stem == pathlib.Path(img).stem, annotations))
    if len(anno):
        anno = cv2.imread(anno[0].as_posix(), -1)
        return anno
    else:
        return None


def save_bin_mask(arr: np.ndarray, save_path: pathlib.Path, stem: str, npy: bool = False, png: bool = False) -> None:
    """
    Save a binary mask as npy or png files.

    Args:
        arr (np.ndarray): Binary mask.
        save_path (pathlib.Path): Path to save the binary mask.
        stem (str): The stem of the file.
        npy (bool): Whether to save as npy file.
        png (bool): Whether to save as png file.
    """
    if npy:
        np.save(save_path / (stem + '.npy'), arr)
    if png:
        cv2.imwrite((save_path / (stem + '.png')).as_posix(), arr.astype(np.uint8) * 255)


def print_eval_results(stats: Union[DictDepthStats, DictGroupMainObjStats], with_main_object: bool = False) -> None:
    """
    Print the evaluation results.
    """
    if not with_main_object:
        for d, stat in stats.items():
            for g, s in stat.items():
                if s.get_count() == 0:
                    continue
                print(f"Group: {g}, Depth: {d}")
                print(s.average)
                print("----")
            print("------------")
    else:
        for g in stats.keys():
            if stats[g].get_count() == 0:
                continue
            print(f"Group: {g}")
            print(stats[g].average)
            print("----")
    print("\n")
