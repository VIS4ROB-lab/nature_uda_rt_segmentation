
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (640, 640)
sim_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize',
         img_scale=[(1280, 960), (1440, 1080), (1600, 1200), (1760, 1320), (1920, 1440), (2080, 1560)],
         multiscale_mode='value', ),
    dict(type='CopyPaste', crop_source='../dataset/building_crops', crop_extension='png',
         crop_label=5, max_pastes=2, prob=0.8, hflip_prob=0.5, over_layer_indices=[0, 1], activation_iter=400, hratio_range=(0.12, 0.4)),
    # if you want to run the training, make sure to download the building crops!
    # if not, make sure to comment/remove this
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.8),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
real_train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    dict(type='Resize',
         img_scale=[(1350, 900), (1500, 1000), (1650, 1100), (1800, 1200), (1950, 1300), (2100, 1400), (2250, 1500)],
         multiscale_mode='value'),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 640),  # (1024, 512)
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type='UDADataset',
        source=dict(
            type='AppleFarmSimDataset',
            data_root='data/Apple_Farm_Sim_winter/',
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=sim_train_pipeline),
        target=dict(
            type='AppleFarmRealDataset',
            data_root='data/andreas-park-split/',
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=real_train_pipeline)),
    # val=dict(
    #     type='AppleFarmRealDataset',
    #     data_root='data/Apple_Farm_Real/'
    #     img_dir='images/validation',
    #     ann_dir='annotations/validation',
    #     pipeline=test_pipeline),
    val=dict(
        type='AppleFarmSimDataset',
        data_root='data/andreas-park-split/',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type='AppleFarmRealDataset',
        data_root='data/andreas-park-split/',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))
