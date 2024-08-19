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
                           num_sample_points=12,
                           negative_cls_id=[classes['terrain'], classes['sky']]),
                RefineTask(foreground_id=classes['terrain'],
                           type=RefineType.REMOVE_HOLES,
                           ignore_top_y=0.3),
                RefineTask(foreground_id=classes['canopy'],
                           type=RefineType.REMOVE_HOLES,
                           negative_cls_id=[classes['terrain'], classes['others'], classes['trunk']]),
                RefineTask(
                    foreground_id=classes['trunk'],
                    type=RefineType.BOUNDARY_SHARPENING,
                    erosion_cfg=ErosionConfig(5, 3, 3, 2),
                    negative_cls_id=classes['canopy'],
                    num_sample_points=9,
                    post_processor={PostProcessorType.INSTANCE: (yshape_post_processor,
                                                                 {"width_change_threshold": 0.35, 'strict': True})},
                    allowed_aspect_ratio=(0.1, 1.25),
                    max_expansion_ratio=500,
                    allowed_connected_components=1,
                ),
                RefineTask(
                    foreground_id=classes['unlabeled'],
                    type=RefineType.LABEL_RECOVERY,
                ),
                ]
