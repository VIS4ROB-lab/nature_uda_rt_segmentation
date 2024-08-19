# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich & UCY V4RL. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from uda_helpers.io import read_yaml
from uda_helpers.refine import RefineTask, RefineType, PostProcessorType, yshape_post_processor, ErosionConfig


_yaml_content = read_yaml('./dataset/classes.yaml', classes=True, palettes=False, pop_unlabeled=False)
classes = _yaml_content['classes']

refine_tasks = [RefineTask(foreground_id=classes['building'],
                           type=RefineType.REGION,
                           num_sample_points=6, negative_cls_id=[classes['terrain']]),
                RefineTask(foreground_id=classes['terrain'],
                           type=RefineType.REGION, ignore_top_y=0.25),
                RefineTask(
                    foreground_id=classes['trunk'],
                    type=RefineType.BOUNDARY_SHARPENING,
                    negative_cls_id=classes['canopy'],
                    num_sample_points=3,
                    post_processor={PostProcessorType.INSTANCE: (yshape_post_processor,
                                                                 {"width_change_threshold": 0.4})},
                    allowed_aspect_ratio=(0, 1.25),
                    allowed_connected_components=1,
                ),
                RefineTask(
                    foreground_id=classes['unlabeled'],
                    type=RefineType.LABEL_RECOVERY,
                    erosion_cfg=ErosionConfig(9, 4, 3, 0),
                    num_sample_points=4,
                    max_expansion_ratio=12.5,
                ),
                RefineTask(
                    foreground_id=classes['canopy'],
                    type=RefineType.REMOVE_HOLES,
                    negative_cls_id=[classes['terrain'], classes['others']]
                )
                ]