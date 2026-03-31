# OVSegFormer 配置文件 — ResNet50 骨干 + Xenium 空间转录组数据

# ============================================================
# 1. 基础实验设置
# ============================================================
experiment_name = 'OVSegFormer_Res50_Xenium'
work_dir = f'./work_dirs/{experiment_name}'  # 日志和权重保存目录
gpu_id = '0'
lr = 1e-4
max_epochs = 5

# ============================================================
# 2. 数据集设置 — Xenium 空间转录组数据
# ============================================================
dataset_type = 'OVSegDataset'
data_root = '/home/lty/AVSegFormer-master copy/AVSegFormer-master/raw_data'
crop_size = 512        # 裁剪 patch 大小 (像素)
pixel_size = 0.2125    # Xenium 像素分辨率 (µm/pixel)

dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split='train',
        crop_size=crop_size,
        pixel_size=pixel_size,
        batch_size=2,                  # 每 batch 的样本数
        num_samples=128,               # 每 epoch 采样 crop 数
        max_transcripts=768,           # 每个 crop 最大转录本数
        max_query_transcripts=64,      # 每个 query (细胞) 最大关联转录本数
        deterministic_eval=False,      # 训练集使用随机裁剪
        split_ratios=(0.8, 0.1, 0.1), # train/val/test 比例
        enforce_disjoint_cells=True,   # 确保不同 split 的细胞不重叠
        split_mode='spatial_x',       # 按 X 坐标空间划分 (避免空间泄露)
        tx_assign_mode='polygon',     # 使用多边形命中检测分配转录本到细胞
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
        max_query_transcripts=64,
        benchmark_seed=123,
        deterministic_eval=True,       # 验证集使用确定性裁剪 (可复现)
        split_ratios=(0.8, 0.1, 0.1),
        enforce_disjoint_cells=True,
        split_mode='spatial_x',
        tx_assign_mode='polygon',
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
        max_query_transcripts=64,
        benchmark_seed=1123,
        deterministic_eval=True,
        split_ratios=(0.8, 0.1, 0.1),
        enforce_disjoint_cells=True,
        split_mode='spatial_x',
        tx_assign_mode='polygon',
    )
)

# ============================================================
# 3. 模型架构设置 — OVSegFormer
# ============================================================
model = dict(
    type='AVSegFormer',
    backbone=dict(
        type='resnet50',
        init_weights_path='torchvision',  # 使用 torchvision 预训练权重
    ),
    # 组学分支参数
    gene_vocab_size=20000,     # 基因词汇表大小 (覆盖所有 codeword_index)
    gene_embed_dim=32,         # 基因嵌入维度
    omics_scalar_dim=1,        # 标量特征维度 (质量值 qv)
    query_graph_k=8,           # 组学图 kNN 的 k 值
    head=dict(
        type='AVSegHead',
        in_channels=[256, 512, 1024, 2048],  # ResNet50 各层输出通道数
        num_classes=1,                        # 二分类 (细胞 vs 背景)
        embed_dim=256,                        # 统一嵌入维度
        num_layers=3,                         # Transformer 解码器层数
        num_heads=8,                          # 多头注意力头数
        multi_scale_indices=[1, 2, 3],        # 使用 ResNet layer2/3/4 的特征
        local_visual_radius=0.18,             # 局部视觉注意力半径 (归一化坐标)
        local_visual_soft_bias_scale=2.0,     # 局部视觉软约束强度
        local_visual_pool_radius=0.18,        # 局部视觉池化半径 (归一化坐标)
        local_visual_pool_type='mean',        # 局部视觉池化方式
        # 核引导 Query 生成器: 质心坐标 → query embedding
        query_generator=dict(
            type='NucleiGuidedQueryGenerator',
            input_dim=2,        # (x, y) 坐标
            embed_dim=256       # 与 Transformer d_model 对齐
        ),
        # Transformer 解码器: 跨模态融合 (视觉 + 组学)
        transformer=dict(
            type='TransformerDecoder',
            d_model=256,
            nhead=8,
            num_encoder_layers=0,    # 不使用编码器 (视觉特征直接作为 memory)
            num_decoder_layers=3,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
        ),
    ),
)

# ============================================================
# 4. 优化器设置
# ============================================================
optimizer = dict(
    type='AdamW',
    lr=lr,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # 骨干网络学习率降为 1/10
        }
    )
)

# ============================================================
# 5. 损失函数设置
# ============================================================
# 语义级损失 (原 AVSegFormer 保留)
loss = dict(
    type='IouSemanticAwareLoss',
    bce_weight=1.0,
    iou_weight=1.0,
    dice_weight=0.5
)

# 实例级损失 (OVSegFormer 核心损失)
instance_loss = dict(
    type='AlignedInstanceSegLoss',
    cls_weight=0.0,               # 分类损失权重 (初始关闭，后续可调)
    mask_bce_weight=1.0,          # 掩膜 BCE 损失权重
    mask_dice_weight=1.0,         # 掩膜 Dice 损失权重
    overlap_weight=0.1,           # 重叠惩罚权重
    no_object_weight=0.1,         # 背景类损失权重
    matcher_cls_cost=0.0,         # Hungarian 匹配器分类代价
    matcher_mask_cost=1.0,        # Hungarian 匹配器掩膜代价
    matcher_dice_cost=1.0,        # Hungarian 匹配器 Dice 代价
    mask_loss_size=128,           # 掩膜损失计算时的下采样尺寸 (节省显存)
    overlap_loss_size=64          # 重叠损失计算时的下采样尺寸
)

# ============================================================
# 6. 训练进程设置
# ============================================================
process = dict(
    num_works=4,                  # DataLoader 工作进程数
    pin_memory=True,              # 固定内存加速 CPU→GPU 传输
    persistent_workers=False,
    prefetch_factor=1,
    log_interval=10,              # 每 N 步打印日志
    diagnostic_interval=5,        # 每 N epoch 运行诊断
    train_epochs=max_epochs,
    freeze_epochs=0,              # 冻结骨干的 epoch 数 (0 = 不冻结)
    val_epochs=1,                 # 每 N epoch 做验证
    display_iter=10
)

# 混合精度训练设置 (针对 RTX 4090D 优化)
use_amp = True
amp_dtype = 'bfloat16'           # 4090D 原生支持 bfloat16
grad_clip_max_norm = 1.0         # 梯度裁剪最大范数
grad_norm_type = 2.0
instance_use_cls_score = False   # 后处理时不使用分类分数 (仅用掩膜)

# 学习率调度器
lr_scheduler = dict(
    type='CosineAnnealingLR',
    t_max=max_epochs,
    eta_min=1e-6,                 # 最小学习率
)

# 过拟合检测模式 (调试用: 用少量数据反复训练验证 loss 是否能下降)
sanity_overfit = dict(
    enabled=True,
    num_samples=16,
    repeat_factor=25,
    val_samples=16,
)

# ============================================================
# 7. 实例评估阈值
# ============================================================
metric_threshold = 0.5
query_score_threshold = 0.5      # query objectness 分数阈值
instance_mask_threshold = 0.5    # 掩膜二值化阈值
instance_match_iou_threshold = 0.5  # 实例匹配 IoU 阈值
instance_overlap_threshold = 0.5
instance_min_area = 16           # 最小实例面积 (像素)
instance_top_k = None            # 保留 top-K 个实例 (None = 全部)
diagnostic_max_cases = 50        # 诊断可视化最大样本数
diagnostic_query_topk = 5        # 诊断时展示 top-K query
