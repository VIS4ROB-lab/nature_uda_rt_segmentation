"""
Core script for refining the predictions using SAM. Read the project README and call "--help" for more details.
"""

# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import importlib
import pathlib
import argparse
from typing import Optional, List
from copy import deepcopy

import torch
import cv2
import numpy as np
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import remove_small_regions

from uda_helpers import save_bin_mask, annotate_points
from uda_helpers.refine import *


class PseudoLabelRefiner:
    """
    SAM-aided Refinement for pseudo-labels.

    Args:
        sam_predictor (SamPredictor): The SAM predictor.
        refine_tasks (List[RefineTask]): The refine tasks.
        device (str): The device to use for inference.
        unlabeled_id (int): The index(id) of the unlabeled class.
        max_expansion_ratio (float): The maximum expansion ratio of identified object. Can be overridden by the task's setting.
        verbose (VerboseLevel): Verbose level.
        verbose_output_dir (Optional[pathlib.Path]): The directory to save the verbose output.
    """
    def __init__(
            self,
            sam_predictor: SamPredictor,
            device: str = "cuda",
            refine_tasks: List[RefineTask] = None,
            unlabeled_id: int = 255,
            max_expansion_ratio: float = 150.,
            verbose: VerboseLevel = VerboseLevel.NONE,
    ):
        self.sam_predictor = sam_predictor
        self.device = device
        self.refine_tasks = refine_tasks
        self.unlabeled_id = unlabeled_id
        self.max_expansion_ratio = max_expansion_ratio
        self.verbose = verbose
        self.verbose_output_dir: Optional[pathlib.Path] = None

        self.default_erosion_cfg = {
            RefineType.REGION: ErosionConfig(7, 4, 3, 3),
            RefineType.BOUNDARY_SHARPENING: ErosionConfig(9, 4, 3, 4),
            RefineType.LABEL_RECOVERY: ErosionConfig(12, 6),
        }

        self.refined_instance = RefinedInstance()

    @staticmethod
    def _save_points_samples(mask: np.ndarray, points: List[List[Tuple]], save_dir: pathlib.Path, fname: str,
                             radius: int = 6) -> None:
        """
        Save the annotated mask with the points samples.

        Args:
            mask (np.ndarray): The mask.
            points (List[List[Tuple]]): The points samples.
            save_dir (pathlib.Path): The save directory.
            fname (str): The file name.
            radius (int): The radius of the annotated points.
        """
        annotated_mask = deepcopy(mask)
        for instance in points:
            annotated_mask = annotate_points(annotated_mask, instance, color=None, radius=radius)
            if save_dir is not None:
                cv2.imwrite((save_dir / fname).as_posix(), annotated_mask)

    def _instance_reset(self) -> None:
        """
        Reset the refined instance.
        """
        del self.refined_instance
        self.refined_instance = RefinedInstance()

    def _verbose_save_mask_and_samples(self, instances: List[List[Tuple]],
                                       task: RefineTask, t: int,
                                       fg_mask_eroded: np.ndarray,
                                       radius: int = 6):
        """
        Save the mask and the points samples in verbose mode.

        Args:
            instances (List[List[Tuple]]): The instances.
            task (RefineTask): The refine task.
            t (int): The task index.
            fg_mask_eroded (np.ndarray): The eroded mask.
            radius (int): The radius of the annotated points.
        """
        if self.verbose == VerboseLevel.ALL and self.verbose_output_dir is not None:
            print(f"Found {len(instances)} instances.")
            save_bin_mask(fg_mask_eroded, self.verbose_output_dir, f"{t}_{task.type.value}_fg_mask_eroded",
                          npy=False, png=True)
            self._save_points_samples(fg_mask_eroded, instances, self.verbose_output_dir,
                                      f"{t}_{task.type.value}_annotated_mask.png", radius=radius)

    def _input_preprocess(self, img_path: pathlib.Path, pred_path: pathlib.Path) -> None:
        """
        Preprocess the input image and prediction.

        Args:
            img_path (pathlib.Path): The image path.
            pred_path (pathlib.Path): The prediction path.
        """
        if not isinstance(self.verbose, VerboseLevel):
            self.verbose = VerboseLevel.NONE
            print(f"Invalid verbose level: {self.verbose}. Reset to None")
        if self.verbose != VerboseLevel.NONE:
            print(f"\n-------------------- Refining {img_path.stem} --------------------")
        self.refined_instance.img = cv2.imread(img_path.as_posix(), -1)
        self.refined_instance.img = cv2.cvtColor(self.refined_instance.img, cv2.COLOR_BGR2RGB)
        if pred_path.suffix == '.npy':
            self.refined_instance.pred = np.load(pred_path.as_posix())
        elif pred_path.suffix == '.png':
            self.refined_instance.pred = cv2.imread(pred_path.as_posix(), -1)
        else:
            raise ValueError(f"Invalid prediction file: {pred_path}!")
        if self.refined_instance.img.shape[:2] != self.refined_instance.pred.shape[:2]:
            self.refined_instance.pred = cv2.resize(
                self.refined_instance.pred,
                dsize=(self.refined_instance.img.shape[1], self.refined_instance.img.shape[0]),
                interpolation=cv2.INTER_NEAREST)

        # check whether the prediction is a semantic map
        if len(self.refined_instance.pred.shape) != 2:
            raise ValueError(f"Invalid prediction shape: {self.refined_instance.pred.shape}! Should be 2D.")

        self.sam_predictor.set_image(self.refined_instance.img)
        self.refined_instance.new_pred = deepcopy(self.refined_instance.pred)

    def _task_preprocess(self, t: int, task: RefineTask) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Common preprocessing for the refinement tasks.

        Args:
            t (int): The task index.
            task (RefineTask): The refine task.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
            The foreground binary mask, eroded binary mask, new blank foreground mask, negative-class(es) mask.
        """
        if self.verbose == VerboseLevel.ALL:
            print(task.type.value)
        if task.type == RefineType.LABEL_RECOVERY:
            task.foreground_id = self.unlabeled_id
        elif task.foreground_id is None:
            raise ValueError(f"Foreground id is not set for task {task}!")

        fg_mask = self.refined_instance.new_pred == task.foreground_id
        fg_new = np.zeros_like(fg_mask)

        if self.verbose == VerboseLevel.ALL and self.verbose_output_dir is not None:
            save_bin_mask(fg_mask, self.verbose_output_dir, f"{t}_{task.type.value}_fg_mask", npy=True, png=True)

        if task.ignore_top_y is not None:
            if isinstance(task.ignore_top_y, float):
                if 0 <= task.ignore_top_y <= 1:
                    ignore_top_y = int(task.ignore_top_y * self.refined_instance.img.shape[0])
                else:
                    raise ValueError(f"Invalid ignore_top_y: {task.ignore_top_y}! Should be in range [0, 1].")
            elif isinstance(task.ignore_top_y, int):
                ignore_top_y = task.ignore_top_y
            else:
                raise ValueError(f"Invalid ignore_top_y: {task.ignore_top_y}! Should be float or int.")
            ignored_mask = np.concatenate((np.zeros((ignore_top_y, self.refined_instance.img.shape[1]), dtype=bool),
                                           np.ones((self.refined_instance.img.shape[0] - ignore_top_y,
                                                    self.refined_instance.img.shape[1]), dtype=bool)),
                                          axis=0)
            fg_mask = fg_mask & ignored_mask

        neg_mask = None
        if task.negative_cls_id is not None:
            if isinstance(task.negative_cls_id, int):
                neg_mask = self.refined_instance.new_pred == task.negative_cls_id
            elif isinstance(task.negative_cls_id, list) and len(task.negative_cls_id) > 0:
                neg_mask = np.isin(self.refined_instance.new_pred, task.negative_cls_id)
            else:
                raise ValueError(f"Invalid negative class id: {task.negative_cls_id}!")
        if neg_mask is not None and self.verbose == VerboseLevel.ALL and self.verbose_output_dir is not None:
            save_bin_mask(neg_mask, self.verbose_output_dir, f"{t}_{task.type.value}_neg_mask", npy=True, png=True)

        self.exp_limit = self.max_expansion_ratio if task.max_expansion_ratio is None else task.max_expansion_ratio

        if task.type != RefineType.REMOVE_HOLES:
            erosion_cfg = task.erosion_cfg if task.erosion_cfg is not None else self.default_erosion_cfg[task.type]
            fg_mask_eroded = shrink(fg_mask, **erosion_cfg.as_dict())
        else:
            fg_mask_eroded = None

        return fg_mask, fg_mask_eroded, fg_new, neg_mask

    def _infer_instances(self, ins: List[List[Tuple]]) -> np.ndarray:
        """
        Group inference for the instance points with SAM.

        Args:
            ins (List[List[Tuple]]): The instance points.

        Returns:
            np.ndarray: The inferred masks.
        """
        pt_coords = np.array(ins).reshape(len(ins), -1, 2)
        tf_pt_coords = self.sam_predictor.transform.apply_coords_torch(torch.from_numpy(pt_coords).to(self.device),
                                                                       self.refined_instance.img.shape[:2])
        pt_labels = torch.ones(pt_coords.shape[0], pt_coords.shape[1]).to(self.device)
        inferred_masks, _, _ = self.sam_predictor.predict_torch(point_coords=tf_pt_coords, point_labels=pt_labels,
                                                                multimask_output=True)
        inferred_masks_ = inferred_masks.cpu().numpy().astype(bool)
        return inferred_masks_

    def _pred_fusion(self, input_pred: np.ndarray, mask: np.ndarray, label: int) -> np.ndarray:
        """
        Prediction fusion.

        Args:
            input_pred (np.ndarray): The input prediction.
            mask (np.ndarray): The mask.
            label (int): The label index to overwrite with the mask.

        Returns:
            np.ndarray: The new prediction.
        """
        new_pred = deepcopy(input_pred)
        new_pred[mask] = label
        return new_pred

    def _region_correction(self, fg_mask_eroded: np.ndarray, fg_new: np.ndarray, neg_mask: np.ndarray,
                           task: RefineTask, t: int) -> np.ndarray:
        """
        The region correction task.

        Args:
            fg_mask_eroded (np.ndarray): The eroded foreground binary mask.
            fg_new (np.ndarray): The new mask for correction.
            neg_mask (np.ndarray): The negative mask.
            task (RefineTask): The refine task.
            t (int): The task index.

        Returns:
            np.ndarray: The corrected mask.
        """
        instances, _ = split_instances(fg_mask_eroded,
                                       min_area=task.min_valid_area if task.min_valid_area is not None and task.min_valid_area >= 0 else 3600,
                                       num_sample_points=task.num_sample_points)
        self._verbose_save_mask_and_samples(instances, task, t, fg_mask_eroded, radius=15)
        if len(instances):
            instances = np.array(instances)
            for i, instance in enumerate(instances):
                if all([fg_new[y, x] for x, y in instance]):
                    continue
                point_coords = np.array(instance)
                point_labels = np.ones((point_coords.shape[0],)).astype(int)
                mask, _, _ = self.sam_predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                                        multimask_output=False)
                mask = mask.squeeze(0)
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                        kernel=np.ones((7, 7), np.uint8), iterations=2).astype(bool)
                if self.verbose == VerboseLevel.ALL and self.verbose_output_dir is not None:
                    save_bin_mask(mask, self.verbose_output_dir, f"{t}_{task.type.value}_{i}_mask", npy=False, png=True)
                fg_new[mask] = True
            if self.verbose == VerboseLevel.ALL and self.verbose_output_dir is not None:
                save_bin_mask(fg_new, self.verbose_output_dir, f"{t}_{task.type.value}_mask_before_neg",
                              npy=False, png=True)
            if neg_mask is not None:
                fg_new = fg_new & neg_mask
            fg_new, status = remove_small_regions(fg_new, area_thresh=2000, mode='holes')
        return fg_new

    def _boundary_sharpening(self, fg_mask_eroded: np.ndarray, fg_new: np.ndarray, neg_mask: np.ndarray,
                             task: RefineTask, t: int) -> np.ndarray:
        """
        The boundary sharpening task.

        Args:
            fg_mask_eroded (np.ndarray): The eroded foreground binary mask.
            fg_new (np.ndarray): The new mask for correction.
            neg_mask (np.ndarray): The negative mask.
            task (RefineTask): The refine task.
            t (int): The task index.

        Returns:
            np.ndarray: The corrected mask.
        """
        instances, ori_masks = split_instances(fg_mask_eroded, min_area=40, num_sample_points=task.num_sample_points)
        self._verbose_save_mask_and_samples(instances, task, t, fg_mask_eroded)
        if len(instances):
            masks_ = self._infer_instances(instances)
            final_masks = []
            for i, ms in enumerate(masks_):
                selected_mask_idx, comp_mask = select_mask(ms, neg_mask, allowed_connected_components=2,
                                                           cov_threshold=0.)

                selected_mask = (ms[selected_mask_idx] & comp_mask) if comp_mask is not None else ms[
                    selected_mask_idx]
                final_mask = selected_mask
                if task.post_processor is not None and PostProcessorType.INSTANCE in task.post_processor:
                    pp = task.post_processor[PostProcessorType.INSTANCE]
                    pp[0](final_mask, **pp[1])
                final_mask = pop_top_n_largest_components(n=task.allowed_connected_components, bin_map=final_mask,
                                                          keep_largest=False)
                if np.sum(final_mask) / np.sum(ori_masks[i]) <= self.exp_limit:
                    final_masks.append(final_mask)
                    if self.verbose == VerboseLevel.ALL and self.verbose_output_dir is not None:
                        vb_out = {'ori_mask': ori_masks[i], 'selected_mask': selected_mask,
                                  'final_mask': final_mask, 'candidate_masks': ms}
                        np.savez_compressed(
                            (self.verbose_output_dir / f"{t}_{task.type.value}_{i}_masks.npz").as_posix(),
                            **vb_out)
            if len(final_masks):
                fg_new = np.bitwise_or.reduce(final_masks, axis=0)
                if task.allowed_aspect_ratio is not None:
                    fg_new = filter_aspect_ratio(fg_new, *task.allowed_aspect_ratio)
            if task.post_processor is not None and PostProcessorType.GLOBAL in task.post_processor:
                pp = task.post_processor[PostProcessorType.GLOBAL]
                fg_new = pp[0](fg_new, **pp[1])
            fg_new, _ = remove_small_regions(fg_new, area_thresh=100, mode='holes')
            return fg_new

    def _label_recovery(self, fg_mask_eroded: np.ndarray, task: RefineTask, t: int):
        """
        The label recovery task. Performs overwriting directly.

        Args:
            fg_mask_eroded (np.ndarray): The eroded foreground binary mask.
            task (RefineTask): The refine task.
            t (int): The task index.

        Returns: None
        """
        instances, ori_masks = split_instances(fg_mask_eroded, min_area=100, num_sample_points=task.num_sample_points)
        self._verbose_save_mask_and_samples(instances, task, t, fg_mask_eroded)
        if len(instances):
            masks_ = self._infer_instances(instances)
            # fill the new instance area with the majority class from the original prediction
            for i, ms in enumerate(masks_):
                selected_mask_idx, _ = select_mask(ms, ~ori_masks[i], cov_threshold=1.)
                if selected_mask_idx == -1:
                    continue
                selected_mask = ms[selected_mask_idx]
                if self.verbose == VerboseLevel.ALL and self.verbose_output_dir is not None:
                    vb_out = {'ori_mask': ori_masks[i], 'selected_mask': selected_mask, 'candidate_masks': ms}
                    np.savez_compressed((self.verbose_output_dir / f"{t}_{task.type.value}_{i}_masks.npz").as_posix(),
                                        **vb_out)
                majority_class = np.argmax(np.bincount(self.refined_instance.pred[ori_masks[i]]))
                if np.sum(selected_mask) / np.sum(ori_masks[i]) > self.exp_limit:
                    self.refined_instance.new_pred[selected_mask & ori_masks[i]] = majority_class
                else:
                    self.refined_instance.new_pred[selected_mask] = majority_class
            if self.verbose != VerboseLevel.NONE and self.verbose_output_dir is not None:
                cv2.imwrite((self.verbose_output_dir / f"{t}_{task.type.value}_pred.png").as_posix(),
                            self.refined_instance.new_pred.astype(np.uint8))

    def _remove_holes(self, fg_mask: np.ndarray, neg_mask: np.ndarray) -> np.ndarray:
        """
        Remove the holes in the mask.

        Args:
            fg_mask (np.ndarray): The foreground mask.
            neg_mask (np.ndarray): The negative mask to limit the removal.

        Returns:
            np.ndarray: The mask without holes.
        """
        fg_closed = cv2.morphologyEx(fg_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                     kernel=np.ones((7, 7), np.uint8), iterations=2).astype(bool)
        fg_new = remove_small_regions(fg_closed, area_thresh=400, mode='holes')[0]
        if neg_mask is not None:
            fg_new = fg_new & neg_mask
        return fg_new

    def single_img_refine(
            self,
            img_path: pathlib.Path,
            pred_path: pathlib.Path,
    ) -> np.ndarray:
        """
        Interface for refining a single image.

        Args:
            img_path (pathlib.Path): The image path.
            pred_path (pathlib.Path): The prediction path.

        Returns:
            np.ndarray: The refined prediction.
        """
        self._instance_reset()
        self._input_preprocess(img_path, pred_path)
        for t, task in enumerate(self.refine_tasks):
            fg_mask, fg_mask_eroded, fg_new, neg_mask = self._task_preprocess(t, task)

            if task.type == RefineType.REGION:
                fg_new = self._region_correction(fg_mask_eroded, fg_new, neg_mask, task, t)
                self.refined_instance.new_pred = self._pred_fusion(self.refined_instance.new_pred, fg_new, task.foreground_id)

            elif task.type == RefineType.BOUNDARY_SHARPENING:
                fg_new = self._boundary_sharpening(fg_mask_eroded, fg_new, neg_mask, task, t)
                self.refined_instance.new_pred = self._pred_fusion(self.refined_instance.new_pred, fg_mask, self.unlabeled_id)
                self.refined_instance.new_pred = self._pred_fusion(self.refined_instance.new_pred, fg_new, task.foreground_id)

            elif task.type == RefineType.LABEL_RECOVERY:
                self._label_recovery(fg_mask_eroded, task, t)

            elif task.type == RefineType.REMOVE_HOLES:
                fg_new = self._remove_holes(fg_mask, neg_mask)
                self.refined_instance.new_pred = self._pred_fusion(self.refined_instance.new_pred, fg_new, task.foreground_id)

            if self.verbose != VerboseLevel.NONE and self.verbose_output_dir is not None:
                if self.verbose == VerboseLevel.ALL:
                    save_bin_mask(fg_new, self.verbose_output_dir, f"{t}_{task.type.value}_fg_new", npy=True, png=True)
                cv2.imwrite((self.verbose_output_dir / f"{t}_{task.type.value}_pred.png").as_posix(), self.refined_instance.new_pred.astype(np.uint8))

        return self.refined_instance.new_pred

    def update_verbose_output_dir(self, verbose_output_dir: pathlib.Path) -> None:
        self.verbose_output_dir = verbose_output_dir


def _get_refine_tasks_names() -> List[str]:
    from tasks import get_task_names
    return get_task_names()


def _int2verbose_level(v: Union[int, str, VerboseLevel]) -> VerboseLevel:
    if v.isdigit():
        v = int(v)
        for vl in VerboseLevel:
            if vl.value == v:
                return vl
    elif isinstance(v, VerboseLevel):
        return v

    else:
        try:
            return VerboseLevel[v.upper()]
        except KeyError:
            raise argparse.ArgumentTypeError(f"Invalid verbose level: {v}!")

    raise argparse.ArgumentTypeError(f"Invalid verbose level: {v}!")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Use sam to refine the predictions.')

    parser.add_argument_group("SAM")
    parser.add_argument("--sam_checkpoint", type=str, default='./refinements/sam_vit_h_4b8939.pth',
                        help="SAM checkpoint file")
    parser.add_argument("--sam_type", type=str, default='vit_h',
                        choices=['h', 'l', 'b', 'vit_h', 'vit_l', 'vit_b', 'default'],
                        help="SAM type")
    parser.add_argument("--tasks", type=str, default='apple_farm', help="Refinement tasks definitions",
                        choices=_get_refine_tasks_names())
    parser.add_argument("--device", type=str, default="cuda", help="Device used for inference")

    parser.add_argument_group("Data")
    parser.add_argument("--image", type=str, help="Original image directory or file")
    parser.add_argument("--pred", type=str, help="Prediction directory or file. Should be semantic maps.")

    parser.add_argument_group("Output")
    parser.add_argument("--out_dir", type=str, help="Output directory")

    parser.add_argument_group("Misc")
    parser.add_argument("--verbose", default=VerboseLevel.NONE,
                        type=_int2verbose_level,
                        help="verbose level to output middle results")
    parser.add_argument("--verbose_out_dir", type=str, default="",
                        help="verbose output directory. "
                             "If not set, the output directory will be used with a subdirectory 'verbose'.")

    args = parser.parse_args()
    return args


def _import_tasks(task_name: str) -> Optional[List[RefineTask]]:
    try:
        task_module = importlib.import_module(f"tasks.{task_name}")
        return _check_tasks(getattr(task_module, 'refine_tasks'))
    except ModuleNotFoundError:
        print(f"No module named '{task_name}' found.")
        return None
    except AttributeError:
        print(f"Module '{task_name}' does not have a 'refine_tasks' variable.")
        return None


def _check_tasks(tasks: List[RefineTask]) -> List[RefineTask]:
    if not isinstance(tasks, list):
        raise ValueError(f"Invalid tasks: {tasks}!")
    if not all(isinstance(t, RefineTask) for t in tasks):
        raise ValueError(f"Invalid tasks: {tasks}!")
    return tasks


def main(args: argparse.Namespace) -> None:
    sam_checkpoint = args.sam_checkpoint
    sam_type = args.sam_type
    if len(sam_type) == 1:
        sam_type = 'vit_' + sam_type
    device = "cuda" if len(args.device) == 0 else args.device

    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    refine_tasks = _import_tasks(args.tasks)

    img_path = pathlib.Path(args.image)
    if img_path.is_dir():
        image_files = sorted([f for f in img_path.rglob("*.[jJ][pP][gG]")], key=lambda x: x.stem)
    else:
        image_files = [img_path]

    pred_path = pathlib.Path(args.pred)
    if pred_path.is_dir():
        pred_files = sorted([f for f in pred_path.rglob("*.png")], key=lambda x: x.stem)
    else:
        pred_files = [pred_path]

    if len(args.out_dir) == 0:
        out_path = pathlib.Path('./refinements')
    else:
        out_path = pathlib.Path(args.out_dir)
    if not out_path.exists():
        out_path.mkdir(parents=True)

    verbose_out_dir = None if not args.verbose else pathlib.Path(args.verbose_out_dir) if len(args.verbose_out_dir) else out_path / "verbose"

    if args.verbose != VerboseLevel.NONE:
        for tsk in refine_tasks:
            print(tsk.short_repr())

    pseudo_label_refiner = PseudoLabelRefiner(predictor, device=device, refine_tasks=refine_tasks, verbose=args.verbose)

    for im, pd in tqdm(zip(image_files, pred_files), total=len(image_files)):
        if im.stem != pd.stem:
            raise ValueError(f"Image {im.stem} and prediction {pd.stem} do not match!")
        vod = verbose_out_dir / im.stem if (args.verbose != VerboseLevel.NONE and verbose_out_dir is not None) else None
        if isinstance(vod, pathlib.Path):
            vod.mkdir(parents=True, exist_ok=True)
        pseudo_label_refiner.update_verbose_output_dir(vod)
        refined = pseudo_label_refiner.single_img_refine(im, pd)
        cv2.imwrite((out_path / (im.stem + '.png')).as_posix(), refined)


if __name__ == "__main__":
    main(_parse_args())
