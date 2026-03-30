# 文件位置: config/ovseg/OVSegFormer_res50.py

# 1. 基础设置
experiment_name = 'OVSegFormer_Res50_Xenium'
# 建议先把 log 和 checkpoint 放在容易找到的地方
work_dir = f'./work_dirs/{experiment_name}'
gpu_id = '0'
lr = 1e-4
max_epochs = 15

# 2. 数据集设置 (Dataset) - [关键修改] 拆分为 train 和 val
dataset_type = 'OVSegDataset'
data_root = '/home/lty/AVSegFormer-master copy/AVSegFormer-master/raw_data'
crop_size = 512
pixel_size = 0.2125

dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        crop_size=crop_size,
        pixel_size=pixel_size,
        batch_size=2,
        num_samples=128,
        max_transcripts=768,
        deterministic_eval=False,
        split_ratios=(0.8, 0.1, 0.1),
        enforce_disjoint_cells=True,
        split_mode='spatial_x',
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        crop_size=crop_size,
        pixel_size=pixel_size,
        batch_size=2,
        num_samples=32,
        max_transcripts=768,
        benchmark_seed=123,
        deterministic_eval=True,
        split_ratios=(0.8, 0.1, 0.1),
        enforce_disjoint_cells=True,
        split_mode='spatial_x',
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='test',
        crop_size=crop_size,
        pixel_size=pixel_size,
        batch_size=2,
        num_samples=32,
        max_transcripts=768,
        benchmark_seed=1123,
        deterministic_eval=True,
        split_ratios=(0.8, 0.1, 0.1),
        enforce_disjoint_cells=True,
        split_mode='spatial_x',
    )
)

# 3. 模型设置 (Model)
model = dict(
    type='AVSegFormer',
    backbone=dict(
        type='resnet50',
        init_weights_path='torchvision',
    ),
    omics_token_num=8,
    omics_token_heads=8,
    gene_vocab_size=20000,
    gene_embed_dim=32,
    omics_scalar_dim=1,
    head=dict(
        type='AVSegHead',
        in_channels=[256, 512, 1024, 2048],
        num_classes=1,
        embed_dim=256,
        num_layers=3,
        num_heads=8,
        multi_scale_indices=[1, 2, 3],
        query_generator=dict(
            type='NucleiGuidedQueryGenerator',
            input_dim=2,
            embed_dim=256
        ),
        transformer=dict(
            type='TransformerDecoder',
            d_model=256,
            nhead=8,
            num_encoder_layers=0,
            num_decoder_layers=3,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
        ),
    ),
)

# 4. 优化器与Loss设置
optimizer = dict(
    type='AdamW',
    lr=lr,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        }
    )
)

# Loss 配置 (沿用原有的配置，确保 train.py 能读到)
loss = dict(
    type='IouSemanticAwareLoss',
    bce_weight=1.0,
    iou_weight=1.0,
    dice_weight=0.5
)

instance_loss = dict(
    type='AlignedInstanceSegLoss',
    cls_weight=0.0,
    mask_bce_weight=1.0,
    mask_dice_weight=1.0,
    overlap_weight=0.1,
    no_object_weight=0.1,
    matcher_cls_cost=0.0,
    matcher_mask_cost=1.0,
    matcher_dice_cost=1.0,
    mask_loss_size=128,
    overlap_loss_size=64
)

# 进程设置
process = dict(
    num_works=4,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=1,
    log_interval=10,
    diagnostic_interval=5,
    train_epochs=max_epochs,
    freeze_epochs=0, # 不冻结
    val_epochs=1,
    display_iter=10
)

use_amp = True
amp_dtype = 'bfloat16'
grad_clip_max_norm = 1.0
grad_norm_type = 2.0
instance_use_cls_score = False
lr_scheduler = dict(
    type='CosineAnnealingLR',
    t_max=max_epochs,
    eta_min=1e-6,
)
sanity_overfit = dict(
    enabled=False,
    num_samples=16,
    repeat_factor=25,
    val_samples=16,
)
metric_threshold = 0.5
query_score_threshold = 0.5
instance_mask_threshold = 0.5
instance_match_iou_threshold = 0.5
instance_overlap_threshold = 0.5
instance_min_area = 16
instance_top_k = None
diagnostic_max_cases = 50
diagnostic_query_topk = 5
