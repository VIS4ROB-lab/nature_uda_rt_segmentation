_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/apple_farm.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes = 7
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor,
             backbone=dict(init_cfg=dict(_delete_=True,
                                         type='Pretrained',
                                         checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-e1a2eed6.pth',
                                         prefix='backbone'),),
             decode_head=dict(num_classes=num_classes),
             auxiliary_head=[
                 dict(
                     type='FCNHead',
                     in_channels=16,
                     channels=16,
                     num_convs=2,
                     num_classes=num_classes,
                     in_index=1,
                     norm_cfg=norm_cfg,
                     concat_input=False,
                     align_corners=False,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
                 dict(
                     type='FCNHead',
                     in_channels=32,
                     channels=64,
                     num_convs=2,
                     num_classes=num_classes,
                     in_index=2,
                     norm_cfg=norm_cfg,
                     concat_input=False,
                     align_corners=False,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
                 dict(
                     type='FCNHead',
                     in_channels=64,
                     channels=256,
                     num_convs=2,
                     num_classes=num_classes,
                     in_index=3,
                     norm_cfg=norm_cfg,
                     concat_input=False,
                     align_corners=False,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
                 dict(
                     type='FCNHead',
                     in_channels=128,
                     channels=1024,
                     num_convs=2,
                     num_classes=num_classes,
                     in_index=4,
                     norm_cfg=norm_cfg,
                     concat_input=False,
                     align_corners=False,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
             ])
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=20000,
        by_epoch=False,
    )
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
