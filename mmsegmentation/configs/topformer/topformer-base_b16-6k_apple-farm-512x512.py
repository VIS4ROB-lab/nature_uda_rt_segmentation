_base_ = [
    '../_base_/models/topformer_base.py', '../_base_/datasets/apple_farm.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_6k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=7,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                         class_weight=[1., 1., 1., 1.2, 1., 1., 1.])
                     )
)

# Re-config the data sampler.
train_dataloader = dict(batch_size=16, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

runner = dict(type='IterBasedRunner', max_iters=6000)
