_base_ = [
    '../../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../../_base_/models/daformer_sepaspp_mitb0.py',
    # Data Loading
    '../../_base_/datasets/uda_farm_sim_to_real_copypaste_dsall.py',
    # Basic UDA Self-Training
    '../../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../../_base_/schedules/poly10warm.py'
]

# Random Seed
seed = 0
# Modifications to Basic UDA
uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[3, 4],  # for thing-class rather than stuff-class. plant
    imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    # pseudo_weight_ignore_top=15,
    # pseudo_weight_ignore_bottom=120
)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)))
model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.1, 1., 1., 1.075, 1.025, 1.05, 1.],
        )
    )
)
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
gpu_ids = range(1, 2)
runner = dict(type='IterBasedRunner', max_iters=12050)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=12)
evaluation = dict(interval=12050, metric='mIoU', efficient_test=True)
# Meta Information for Result Analysis
name = 'farm_uda_rcs_copypaste_daformer'
exp = 'basic'
name_dataset = 'farm_simreal'
name_architecture = 'daformer_sepaspp_mitb0'
name_encoder = 'mitb0'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_rcs_things'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_b4_12k'
