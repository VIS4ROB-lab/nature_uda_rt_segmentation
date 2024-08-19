_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/apple_farm.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_10k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor, backbone=dict(type='ResNet'), pretrained='torchvision://resnet50',
             decode_head=dict(num_classes=7,
                              loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                                               class_weight=[1., 1., 1., 1.25, 1.025, 1.05, 1.])))
