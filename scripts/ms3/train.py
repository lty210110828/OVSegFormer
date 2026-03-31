import sys
import os
# 将项目根目录加入 python 搜索路径，使 import model / import dataloader 等模块能正确解析
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
import time
import random
import numpy as np
import argparse
from mmcv import Config
from torch.utils.data import DataLoader, Subset, RandomSampler

from utils import pyutils
from utils.loss_util import LossUtil, AlignedInstanceSegLoss, HungarianInstanceSegLoss
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset
from scripts.ms3.utility import (
    compute_aligned_instance_metrics,
    compute_postprocessed_instance_metrics,
    postprocess_instance_predictions,
)


def compute_grad_norm(parameters, norm_type=2.0):
    """计算模型参数梯度的 L_p 范数，用于监控梯度爆炸/消失"""
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return 0.0
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        return float(max(p.grad.detach().abs().max().item() for p in params))
    total = 0.0
    for p in params:
        grad_norm = p.grad.detach().data.norm(norm_type).item()
        total += grad_norm ** norm_type
    return float(total ** (1.0 / norm_type))


def summarize_query_scores(output_cls, query_valid_mask, score_threshold):
    """
    统计 query objectness 分数的分布 (诊断用)

    用于判断分类头是否饱和 (所有分数都接近 1.0 说明阈值失去筛选作用)
    """
    scores = torch.sigmoid(output_cls.squeeze(-1))
    valid_scores = scores[query_valid_mask]
    if valid_scores.numel() == 0:
        return dict(score_mean=0.0, score_std=0.0, score_min=0.0, score_max=0.0, keep_ratio=0.0)
    keep_ratio = float((valid_scores >= score_threshold).float().mean().item()) if score_threshold is not None else 1.0
    return dict(
        score_mean=float(valid_scores.mean().item()),
        score_std=float(valid_scores.std(unbiased=False).item()),
        score_min=float(valid_scores.min().item()),
        score_max=float(valid_scores.max().item()),
        keep_ratio=keep_ratio,  # 超过阈值的 query 比例
    )


def build_scheduler(optimizer, cfg):
    """根据配置构建学习率调度器 (支持 CosineAnnealing / MultiStep / StepLR)"""
    sched_cfg = getattr(cfg, 'lr_scheduler', None)
    if sched_cfg is None:
        return None
    sched_type = str(getattr(sched_cfg, 'type', '')).lower()
    if sched_type == 'cosineannealinglr':
        t_max = int(getattr(sched_cfg, 't_max', getattr(cfg.process, 'train_epochs', 1)))
        eta_min = float(getattr(sched_cfg, 'eta_min', 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    if sched_type == 'multisteplr':
        milestones = list(getattr(sched_cfg, 'milestones', []))
        gamma = float(getattr(sched_cfg, 'gamma', 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    if sched_type == 'steplr':
        step_size = int(getattr(sched_cfg, 'step_size', 10))
        gamma = float(getattr(sched_cfg, 'gamma', 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f'Unsupported lr_scheduler.type: {sched_cfg.type}')


def compute_basic_metrics(pred_logits, target, threshold=0.5, eps=1e-6):
    """计算语义级分割指标 (precision, recall, F1, IoU, Dice)"""
    prob = torch.sigmoid(pred_logits)
    pred = (prob > threshold).float()
    target = target.float()
    dims = tuple(range(1, pred.dim()))
    tp = (pred * target).sum(dims)
    fp = (pred * (1 - target)).sum(dims)
    fn = ((1 - pred) * target).sum(dims)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item(),
        'iou': iou.mean().item(),
        'dice': dice.mean().item()
    }


def compute_omics_coverage(pred_logits, omics_x, mask_size, threshold=0.5):
    """计算转录本被预测掩膜覆盖的比例 (评估预测质量)"""
    prob = torch.sigmoid(pred_logits)
    pred = (prob > threshold).float()
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    h, w = mask_size
    coverages = []
    for b, pts in enumerate(omics_x):
        if pts.numel() == 0:
            continue
        x = (pts[:, 0] * (w - 1)).long().clamp(0, w - 1)
        y = (pts[:, 1] * (h - 1)).long().clamp(0, h - 1)
        inside = pred[b, y, x]
        coverages.append(inside.float().mean().item())
    if len(coverages) == 0:
        return 0.0
    return float(sum(coverages) / len(coverages))


def aggregate_query_masks(output_cls, output_mask, query_valid_mask, query_score_threshold=None, apply_score_threshold=True):
    """
    将多个 query 的掩膜聚合为单一语义分割图

    策略: 每个像素取所有 query 中 score * mask_prob 最大的值
    若某个样本所有 query 都低于阈值，则保留最高分的那个 query (兜底)
    """
    query_scores = torch.sigmoid(output_cls.squeeze(-1))
    query_scores = query_scores.masked_fill(~query_valid_mask, 0.0)
    keep_mask = query_valid_mask
    if apply_score_threshold and query_score_threshold is not None:
        keep_mask = query_valid_mask & (query_scores >= query_score_threshold)
        any_valid = query_valid_mask.any(dim=1)
        no_keep = ~keep_mask.any(dim=1) & any_valid
        if no_keep.any():
            # 兜底: 保留每个样本最高分的 query
            masked_scores = query_scores.masked_fill(~query_valid_mask, -1.0)
            top_idx = masked_scores.argmax(dim=1)
            keep_mask[no_keep, :] = False
            keep_mask[no_keep, top_idx[no_keep]] = True
    mask_prob = torch.sigmoid(output_mask)
    weighted_prob = mask_prob * query_scores[:, :, None, None]
    weighted_prob = weighted_prob.masked_fill(~keep_mask[:, :, None, None], 0.0)
    pred_prob = weighted_prob.max(dim=1, keepdim=True)[0]  # 取每个像素最大值
    pred_prob = pred_prob.clamp_(1e-4, 1.0 - 1e-4)
    return torch.logit(pred_prob)


def custom_collate_fn(batch):
    """
    自定义 batch 整理函数 — 处理变长细胞数和转录本数

    由于每个 crop 包含的细胞数量和每个细胞的转录本数量不同，
    需要将它们 padding 到 batch 内最大值，并生成对应的 valid mask

    核心逻辑:
        1. 细胞维度: pad 到 max_cells，生成 centroid_valid_mask
        2. 转录本维度: pad 到 max_query_tx，生成 query_tx_mask
        3. query_valid_mask = centroid_valid_mask & instance_target_valid
           (仅当质心有效且有有效掩膜时，该 query 才参与训练)
    """
    imgs = torch.stack([item['img'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    centroids = [item['centroids'] for item in batch]
    instance_masks = [item['instance_masks'] for item in batch]
    instance_target_valid = [item['instance_target_valid'] for item in batch]
    query_tx_x = [item['query_tx_x'] for item in batch]
    query_tx_gene_ids = [item['query_tx_gene_ids'] for item in batch]
    query_tx_qv = [item['query_tx_qv'] for item in batch]

    # 计算本 batch 中最大细胞数
    max_cells = max([c.shape[0] for c in centroids])
    if max_cells == 0: max_cells = 1

    padded_centroids = []
    centroid_valid_mask = []
    padded_instance_masks = []
    padded_instance_target_valid = []

    # 计算本 batch 中每个细胞的最大转录本数
    max_query_tx = 1
    for sample_query_tx in query_tx_x:
        for tx in sample_query_tx:
            max_query_tx = max(max_query_tx, int(tx.shape[0]))

    padded_query_tx_x = []
    padded_query_tx_gene_ids = []
    padded_query_tx_qv = []
    padded_query_tx_mask = []

    # 逐样本 padding
    for c in centroids:
        n = c.shape[0]
        idx = len(padded_centroids)
        cur_instance_masks = instance_masks[idx]
        cur_instance_valid = instance_target_valid[idx]
        cur_query_tx_x = query_tx_x[idx]
        cur_query_tx_gene_ids = query_tx_gene_ids[idx]
        cur_query_tx_qv = query_tx_qv[idx]

        if n < max_cells:
            # 细胞数不足 max_cells，用零向量填充
            pad = torch.zeros((max_cells - n, 2), dtype=c.dtype, device=c.device)
            padded_centroids.append(torch.cat([c, pad], dim=0))
            valid = torch.cat([
                torch.ones(n, dtype=torch.bool, device=c.device),
                torch.zeros(max_cells - n, dtype=torch.bool, device=c.device)
            ], dim=0)
            centroid_valid_mask.append(valid)
            mask_pad = torch.zeros((max_cells - n, cur_instance_masks.shape[-2], cur_instance_masks.shape[-1]), dtype=cur_instance_masks.dtype, device=cur_instance_masks.device)
            padded_instance_masks.append(torch.cat([cur_instance_masks, mask_pad], dim=0))
            valid_pad = torch.zeros(max_cells - n, dtype=torch.bool, device=cur_instance_valid.device)
            padded_instance_target_valid.append(torch.cat([cur_instance_valid, valid_pad], dim=0))
        else:
            padded_centroids.append(c)
            centroid_valid_mask.append(torch.ones(max_cells, dtype=torch.bool, device=c.device))
            padded_instance_masks.append(cur_instance_masks)
            padded_instance_target_valid.append(cur_instance_valid)

        # 逐细胞 padding 转录本
        cur_padded_tx_x = []
        cur_padded_tx_gene_ids = []
        cur_padded_tx_qv = []
        cur_padded_tx_mask = []
        for cell_idx in range(max_cells):
            if cell_idx < len(cur_query_tx_x):
                tx_x = cur_query_tx_x[cell_idx]
                tx_gene_ids = cur_query_tx_gene_ids[cell_idx]
                tx_qv = cur_query_tx_qv[cell_idx]
            else:
                # 超出实际细胞数的填充位置
                tx_x = torch.zeros((0, 2), dtype=torch.float32)
                tx_gene_ids = torch.zeros((0,), dtype=torch.long)
                tx_qv = torch.zeros((0, 1), dtype=torch.float32)
            tx_count = int(tx_x.shape[0])
            if tx_count < max_query_tx:
                # 转录本不足，零填充
                tx_x = torch.cat([tx_x, torch.zeros((max_query_tx - tx_count, 2), dtype=tx_x.dtype)], dim=0)
                tx_gene_ids = torch.cat([tx_gene_ids, torch.zeros((max_query_tx - tx_count,), dtype=tx_gene_ids.dtype)], dim=0)
                tx_qv = torch.cat([tx_qv, torch.zeros((max_query_tx - tx_count, 1), dtype=tx_qv.dtype)], dim=0)
            elif tx_count > max_query_tx:
                # 转录本过多，截断
                tx_x = tx_x[:max_query_tx]
                tx_gene_ids = tx_gene_ids[:max_query_tx]
                tx_qv = tx_qv[:max_query_tx]
                tx_count = max_query_tx
            # 生成转录本有效掩膜
            tx_mask = torch.zeros((max_query_tx,), dtype=torch.bool)
            tx_mask[:tx_count] = True
            cur_padded_tx_x.append(tx_x)
            cur_padded_tx_gene_ids.append(tx_gene_ids)
            cur_padded_tx_qv.append(tx_qv)
            cur_padded_tx_mask.append(tx_mask)
        # [max_cells, max_query_tx, ...]
        padded_query_tx_x.append(torch.stack(cur_padded_tx_x, dim=0))
        padded_query_tx_gene_ids.append(torch.stack(cur_padded_tx_gene_ids, dim=0))
        padded_query_tx_qv.append(torch.stack(cur_padded_tx_qv, dim=0))
        padded_query_tx_mask.append(torch.stack(cur_padded_tx_mask, dim=0))

    # Stack 成 batch tensor
    padded_centroids = torch.stack(padded_centroids)                     # [B, max_cells, 2]
    centroid_valid_mask = torch.stack(centroid_valid_mask)               # [B, max_cells]
    padded_instance_masks = torch.stack(padded_instance_masks)           # [B, max_cells, H, W]
    padded_instance_target_valid = torch.stack(padded_instance_target_valid)  # [B, max_cells]
    padded_query_tx_x = torch.stack(padded_query_tx_x)                  # [B, max_cells, max_tx, 2]
    padded_query_tx_gene_ids = torch.stack(padded_query_tx_gene_ids)    # [B, max_cells, max_tx]
    padded_query_tx_qv = torch.stack(padded_query_tx_qv)                # [B, max_cells, max_tx, 1]
    padded_query_tx_mask = torch.stack(padded_query_tx_mask)             # [B, max_cells, max_tx]

    # query 有效条件: 质心存在 且 实例掩膜有效
    query_valid_mask = centroid_valid_mask & padded_instance_target_valid  # [B, max_cells]

    return {
        'img': imgs,
        'mask': masks,
        'query_tx_x': padded_query_tx_x,
        'query_tx_gene_ids': padded_query_tx_gene_ids,
        'query_tx_qv': padded_query_tx_qv,
        'query_tx_mask': padded_query_tx_mask,
        'centroids': padded_centroids,
        'instance_masks': padded_instance_masks,
        'instance_target_valid': padded_instance_target_valid,
        'centroid_valid_mask': centroid_valid_mask,
        'query_valid_mask': query_valid_mask,
    }


def evaluate(model, dataloader, use_amp, amp_dtype, query_score_threshold, instance_mask_threshold, instance_match_iou_threshold, instance_min_area, instance_top_k, use_cls_score):
    """
    验证循环 — 在验证集上计算实例级分割指标

    两种评估方式并行:
        1. PP (Post-Processed): 经过后处理 (阈值筛选/NMS/最小面积过滤) 后的指标
        2. Aligned: 直接用 Hungarian 匹配 query 到 GT (跳过后处理)
    """
    model.eval()
    instance_metrics_list = []
    aligned_instance_metrics_list = []
    with torch.no_grad():
        for batch_data in dataloader:
            imgs = batch_data['img'].cuda(non_blocking=True)
            centroids = batch_data['centroids'].cuda(non_blocking=True)
            query_valid_mask = batch_data['query_valid_mask'].cuda(non_blocking=True)
            instance_masks = batch_data['instance_masks'].cuda(non_blocking=True)
            instance_target_valid = batch_data['instance_target_valid'].cuda(non_blocking=True)
            query_tx_x = batch_data['query_tx_x'].cuda(non_blocking=True)
            query_tx_gene_ids = batch_data['query_tx_gene_ids'].cuda(non_blocking=True)
            query_tx_qv = batch_data['query_tx_qv'].cuda(non_blocking=True)
            query_tx_mask = batch_data['query_tx_mask'].cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                output_cls, output_mask = model(
                    imgs,
                    query_tx_x,
                    centroids,
                    omics_gene_ids=query_tx_gene_ids,
                    omics_qv=query_tx_qv,
                    omics_valid_mask=query_tx_mask,
                    query_valid_mask=query_valid_mask,
                )

            # 后处理实例评估
            batch_instances = postprocess_instance_predictions(
                    output_cls.detach(),
                    output_mask.detach(),
                    query_valid_mask,
                    target_size=instance_masks.shape[-2:],
                    query_score_threshold=query_score_threshold,
                    mask_threshold=instance_mask_threshold,
                    min_area=instance_min_area,
                    top_k=instance_top_k,
                    use_cls_score=use_cls_score,
                )
            instance_metrics = compute_postprocessed_instance_metrics(
                batch_instances,
                instance_masks,
                instance_target_valid,
                match_iou_threshold=instance_match_iou_threshold,
            )
            # Aligned 实例评估 (直接匹配)
            aligned_instance_metrics = compute_aligned_instance_metrics(
                output_cls.detach(),
                output_mask.detach(),
                instance_masks,
                instance_target_valid,
                query_score_threshold=query_score_threshold if use_cls_score else None,
                mask_threshold=instance_mask_threshold,
                match_iou_threshold=instance_match_iou_threshold,
            )
            instance_metrics_list.append(instance_metrics)
            aligned_instance_metrics_list.append(aligned_instance_metrics)

    if len(instance_metrics_list) == 0:
        return {
            'pp_inst_precision': 0.0,
            'pp_inst_recall': 0.0,
            'pp_inst_f1': 0.0,
            'pp_inst_mean_iou': 0.0,
            'pp_inst_matched_iou': 0.0,
            'pp_inst_overlap': 0.0,
            'pp_inst_pred_count': 0.0,
            'pp_inst_gt_count': 0.0,
            'pp_inst_match_count': 0.0,
            'aligned_inst_f1': 0.0,
            'aligned_inst_mean_iou': 0.0,
            'aligned_inst_matched_iou': 0.0,
        }

    # 对所有 batch 的指标取均值
    reduced_metrics = {}
    for key in instance_metrics_list[0].keys():
        reduced_metrics[key] = float(np.mean([m[key] for m in instance_metrics_list]))
    reduced_metrics['aligned_inst_f1'] = float(np.mean([m['inst_f1'] for m in aligned_instance_metrics_list]))
    reduced_metrics['aligned_inst_mean_iou'] = float(np.mean([m['inst_mean_iou'] for m in aligned_instance_metrics_list]))
    reduced_metrics['aligned_inst_matched_iou'] = float(np.mean([m['inst_matched_iou'] for m in aligned_instance_metrics_list]))
    return reduced_metrics


def main():
    """
    OVSegFormer 训练主函数

    训练流程:
        1. 加载配置 → 构建模型 (DataParallel) → 构建数据集和 DataLoader
        2. 初始化实例级损失函数 (AlignedInstanceSegLoss 或 HungarianInstanceSegLoss)
        3. 训练循环: 前向 → 计算损失 → 反向传播 → 梯度裁剪 → 参数更新
        4. 定期验证 → 保存最优模型 (按 PP_InstF1 排序)
        5. 异常中断时紧急保存权重
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('--log_dir', type=str, default='work_dir', help='log dir')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='checkpoint dir')
    args = parser.parse_args()

    # 固定随机种子确保可复现
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)
    # 启用 cuDNN benchmark 和 TF32 以加速 4090D 上的训练
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    cfg = Config.fromfile(args.cfg)

    # 权重保存路径: 优先命令行参数，其次配置文件中的 work_dir
    save_root = args.checkpoint_dir if args.checkpoint_dir else cfg.work_dir
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    logger = getLogger(os.path.join(save_root, 'train.log'), __name__)
    logger.info(f'Config loaded from {args.cfg}')
    logger.info(f'Weights will be saved to: {save_root}')

    # 构建模型并包裹 DataParallel (多卡训练)
    model = build_model(**cfg.model)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    # 构建训练/验证数据集
    train_dataset = build_dataset(**cfg.dataset.train)
    val_dataset = build_dataset(**cfg.dataset.val)

    # 从配置中读取超参数
    batch_size = getattr(cfg.dataset.train, 'batch_size', 16)
    val_batch_size = getattr(cfg.dataset.val, 'batch_size', 1)
    num_workers = getattr(cfg.process, 'num_works', 16)
    pin_memory = getattr(cfg.process, 'pin_memory', True)
    persistent_workers = getattr(cfg.process, 'persistent_workers', True)
    prefetch_factor = getattr(cfg.process, 'prefetch_factor', 2)
    log_interval = getattr(cfg.process, 'log_interval', 10)
    val_interval = max(1, int(getattr(cfg.process, 'val_epochs', 1)))

    # 混合精度设置 (优先 bfloat16，4090D 原生支持)
    use_amp = getattr(cfg, 'use_amp', True)
    amp_dtype_cfg = str(getattr(cfg, 'amp_dtype', 'bfloat16')).lower()
    if amp_dtype_cfg == 'bfloat16' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    # 实例评估阈值
    metric_threshold = getattr(cfg, 'metric_threshold', 0.5)
    query_score_threshold = getattr(cfg, 'query_score_threshold', 0.5)
    instance_mask_threshold = getattr(cfg, 'instance_mask_threshold', metric_threshold)
    instance_match_iou_threshold = getattr(cfg, 'instance_match_iou_threshold', 0.5)
    instance_min_area = int(getattr(cfg, 'instance_min_area', 16))
    instance_top_k = getattr(cfg, 'instance_top_k', None)
    instance_use_cls_score = bool(getattr(cfg, 'instance_use_cls_score', False))

    # 梯度裁剪
    grad_clip_max_norm = float(getattr(cfg, 'grad_clip_max_norm', 0.0))
    grad_norm_type = float(getattr(cfg, 'grad_norm_type', 2.0))

    # 诊断和 sanity overfit 设置
    diag_interval = int(getattr(cfg.process, 'diagnostic_interval', getattr(cfg, 'diagnostic_interval', log_interval)))
    sanity_cfg = getattr(cfg, 'sanity_overfit', None)
    sanity_enabled = bool(getattr(sanity_cfg, 'enabled', False)) if sanity_cfg is not None else False
    sanity_num_samples = int(getattr(sanity_cfg, 'num_samples', 16)) if sanity_cfg is not None else 16
    sanity_repeat_factor = int(getattr(sanity_cfg, 'repeat_factor', 20)) if sanity_cfg is not None else 20
    sanity_val_samples = int(getattr(sanity_cfg, 'val_samples', sanity_num_samples)) if sanity_cfg is not None else sanity_num_samples

    # Sanity overfit 模式: 用少量数据反复训练，验证 loss 能否下降 (调试用)
    if sanity_enabled:
        train_len = len(train_dataset)
        val_len = len(val_dataset)
        train_keep = min(max(1, sanity_num_samples), train_len)
        val_keep = min(max(1, sanity_val_samples), val_len)
        train_dataset = Subset(train_dataset, list(range(train_keep)))
        val_dataset = Subset(val_dataset, list(range(val_keep)))
        train_sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=max(batch_size, train_keep) * max(1, sanity_repeat_factor),
        )
        train_shuffle = False
        logger.info(
            f'SanityOverfit enabled: train_keep={train_keep}, val_keep={val_keep}, '
            f'repeat_factor={sanity_repeat_factor}'
        )
    else:
        train_sampler = None
        train_shuffle = True

    # 构建 DataLoader
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
    )
    if train_sampler is not None:
        loader_kwargs.pop('shuffle', None)
        loader_kwargs['sampler'] = train_sampler
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor
    train_dataloader = DataLoader(train_dataset, **loader_kwargs)
    val_loader_kwargs = dict(
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
    )
    if num_workers > 0:
        val_loader_kwargs['persistent_workers'] = persistent_workers
        val_loader_kwargs['prefetch_factor'] = prefetch_factor
    val_dataloader = DataLoader(val_dataset, **val_loader_kwargs)

    # 优化器、学习率调度器、AMP GradScaler
    optimizer = pyutils.get_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 获取 DataParallel 内部的原始模型引用 (用于保存干净的 state_dict)
    model_without_wrapper = model.module if hasattr(model, 'module') else model

    # 初始化实例级损失函数
    instance_loss_cfg = getattr(cfg, 'instance_loss', None)
    instance_loss_type = str(getattr(instance_loss_cfg, 'type', 'AlignedInstanceSegLoss')) if instance_loss_cfg is not None else 'AlignedInstanceSegLoss'
    instance_cls_weight = getattr(instance_loss_cfg, 'cls_weight', 1.0) if instance_loss_cfg is not None else 1.0
    instance_mask_bce_weight = getattr(instance_loss_cfg, 'mask_bce_weight', 1.0) if instance_loss_cfg is not None else 1.0
    instance_mask_dice_weight = getattr(instance_loss_cfg, 'mask_dice_weight', 1.0) if instance_loss_cfg is not None else 1.0
    instance_overlap_weight = getattr(instance_loss_cfg, 'overlap_weight', 0.0) if instance_loss_cfg is not None else 0.0
    instance_no_object_weight = getattr(instance_loss_cfg, 'no_object_weight', 0.1) if instance_loss_cfg is not None else 0.1
    matcher_cls_cost = getattr(instance_loss_cfg, 'matcher_cls_cost', 1.0) if instance_loss_cfg is not None else 1.0
    matcher_mask_cost = getattr(instance_loss_cfg, 'matcher_mask_cost', 1.0) if instance_loss_cfg is not None else 1.0
    matcher_dice_cost = getattr(instance_loss_cfg, 'matcher_dice_cost', 1.0) if instance_loss_cfg is not None else 1.0
    mask_loss_size = getattr(instance_loss_cfg, 'mask_loss_size', 128) if instance_loss_cfg is not None else 128
    overlap_loss_size = getattr(instance_loss_cfg, 'overlap_loss_size', 64) if instance_loss_cfg is not None else 64

    if instance_loss_type == 'HungarianInstanceSegLoss':
        instance_criterion = HungarianInstanceSegLoss(
            cls_weight=instance_cls_weight,
            mask_bce_weight=instance_mask_bce_weight,
            mask_dice_weight=instance_mask_dice_weight,
            overlap_weight=instance_overlap_weight,
            no_object_weight=instance_no_object_weight,
            matcher_cls_cost=matcher_cls_cost,
            matcher_mask_cost=matcher_mask_cost,
            matcher_dice_cost=matcher_dice_cost,
            mask_loss_size=mask_loss_size,
            overlap_loss_size=overlap_loss_size,
        ).cuda()
    elif instance_loss_type == 'AlignedInstanceSegLoss':
        instance_criterion = AlignedInstanceSegLoss(
            cls_weight=instance_cls_weight,
            mask_bce_weight=instance_mask_bce_weight,
            mask_dice_weight=instance_mask_dice_weight,
            overlap_weight=instance_overlap_weight,
            mask_loss_size=mask_loss_size,
            overlap_loss_size=overlap_loss_size,
        ).cuda()
    else:
        raise ValueError(f'Unsupported instance_loss.type: {instance_loss_type}')
    loss_meter = LossUtil({})

    global_step = 0
    best_inst_f1 = float('-inf')

    # ===== 主训练循环 =====
    try:
        for epoch in range(cfg.process.train_epochs):
            model.train()
            epoch_loss = []

            iter_start = time.perf_counter()
            for n_iter, batch_data in enumerate(train_dataloader):
                data_time = time.perf_counter() - iter_start

                # 数据搬运到 GPU (non_blocking=True 异步传输)
                imgs = batch_data['img'].cuda(non_blocking=True)
                masks = batch_data['mask'].cuda(non_blocking=True)
                centroids = batch_data['centroids'].cuda(non_blocking=True)
                query_valid_mask = batch_data['query_valid_mask'].cuda(non_blocking=True)
                instance_masks = batch_data['instance_masks'].cuda(non_blocking=True)
                instance_target_valid = batch_data['instance_target_valid'].cuda(non_blocking=True)
                query_tx_x = batch_data['query_tx_x'].cuda(non_blocking=True)
                query_tx_gene_ids = batch_data['query_tx_gene_ids'].cuda(non_blocking=True)
                query_tx_qv = batch_data['query_tx_qv'].cuda(non_blocking=True)
                query_tx_mask = batch_data['query_tx_mask'].cuda(non_blocking=True)

                # 前向传播 (混合精度)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    output_cls, output_mask = model(
                        imgs,
                        query_tx_x,
                        centroids,
                        omics_gene_ids=query_tx_gene_ids,
                        omics_qv=query_tx_qv,
                        omics_valid_mask=query_tx_mask,
                        query_valid_mask=query_valid_mask,
                    )
                    # 计算实例级损失
                    instance_loss, instance_loss_dict, _ = instance_criterion(
                        output_cls,
                        output_mask,
                        instance_masks,
                        instance_target_valid,
                    )
                    loss = instance_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Non-finite loss detected: {loss.item()}')
                    loss_dict = dict(instance_loss_dict)

                # 反向传播 + 梯度裁剪 + 参数更新
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm_pre_clip = compute_grad_norm(model.parameters(), norm_type=grad_norm_type)
                if grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm, norm_type=grad_norm_type)
                grad_norm_post_clip = compute_grad_norm(model.parameters(), norm_type=grad_norm_type)
                scaler.step(optimizer)
                scaler.update()

                global_step += 1
                epoch_loss.append(loss.item())
                loss_meter.add_loss(loss, loss_dict)

                # 定期日志输出 (训练指标 + 速度 + 梯度范数)
                iter_time = time.perf_counter() - iter_start
                if global_step % log_interval == 0:
                    imgs_per_sec = batch_size / iter_time if iter_time > 0 else 0.0
                    batch_instances = postprocess_instance_predictions(
                        output_cls.detach(),
                        output_mask.detach(),
                        query_valid_mask,
                        target_size=instance_masks.shape[-2:],
                        query_score_threshold=query_score_threshold if instance_use_cls_score else None,
                        mask_threshold=instance_mask_threshold,
                        min_area=instance_min_area,
                        top_k=instance_top_k,
                        use_cls_score=instance_use_cls_score,
                    )
                    instance_metrics = compute_postprocessed_instance_metrics(
                        batch_instances,
                        instance_masks,
                        instance_target_valid,
                        match_iou_threshold=instance_match_iou_threshold,
                    )
                    aligned_instance_metrics = compute_aligned_instance_metrics(
                        output_cls.detach(),
                        output_mask.detach(),
                        instance_masks,
                        instance_target_valid,
                        query_score_threshold=query_score_threshold if instance_use_cls_score else None,
                        mask_threshold=instance_mask_threshold,
                        match_iou_threshold=instance_match_iou_threshold,
                    )
                    logger.info(
                        f'Epoch: {epoch}, Step: {global_step}, '
                        f'{loss_meter.pretty_out()}'
                        f'PP_InstF1: {instance_metrics["pp_inst_f1"]:.4f}, '
                        f'PP_InstIoU: {instance_metrics["pp_inst_mean_iou"]:.4f}, '
                        f'PP_MatchedIoU: {instance_metrics["pp_inst_matched_iou"]:.4f}, '
                        f'PP_InstP: {instance_metrics["pp_inst_precision"]:.4f}, '
                        f'PP_InstR: {instance_metrics["pp_inst_recall"]:.4f}, '
                        f'PP_Pred: {instance_metrics["pp_inst_pred_count"]:.1f}, '
                        f'PP_GT: {instance_metrics["pp_inst_gt_count"]:.1f}, '
                        f'AlignedInstF1: {aligned_instance_metrics["inst_f1"]:.4f}, '
                        f'AlignedInstIoU: {aligned_instance_metrics["inst_mean_iou"]:.4f}, '
                        f'data_time: {data_time:.4f}, '
                        f'iter_time: {iter_time:.4f}, imgs/s: {imgs_per_sec:.2f}, '
                        f'grad_pre_clip: {grad_norm_pre_clip:.4f}, grad_post_clip: {grad_norm_post_clip:.4f}, '
                        f'lr: {optimizer.param_groups[0]["lr"]:.8f}'
                    )

                # 定期 query 诊断 (检查分类头是否饱和)
                if global_step % diag_interval == 0:
                    score_diag = summarize_query_scores(output_cls.detach(), query_valid_mask, query_score_threshold)
                    mask_prob = torch.sigmoid(output_mask.detach())
                    logger.info(
                        f'Diag Epoch: {epoch}, Step: {global_step}, '
                        f'score_mean: {score_diag["score_mean"]:.4f}, score_std: {score_diag["score_std"]:.4f}, '
                        f'score_min: {score_diag["score_min"]:.4f}, score_max: {score_diag["score_max"]:.4f}, '
                        f'score_keep_ratio: {score_diag["keep_ratio"]:.4f}, '
                        f'mask_prob_mean: {mask_prob.mean().item():.4f}, mask_prob_std: {mask_prob.std(unbiased=False).item():.4f}, '
                        f'mask_logit_abs_mean: {output_mask.detach().abs().mean().item():.4f}'
                    )
                iter_start = time.perf_counter()

            # 每 epoch 结束保存 latest 权重 (保存干净的 state_dict，去除 .module 前缀)
            save_path = os.path.join(save_root, 'latest.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_without_wrapper.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': np.mean(epoch_loss),
            }
            torch.save(checkpoint, save_path)
            logger.info(f'>>> Epoch {epoch} finished. Model saved to {save_path}')

            # 每 10 epoch 保存历史权重
            if (epoch + 1) % 10 == 0:
                history_path = os.path.join(save_root, f'epoch_{epoch+1}.pth')
                torch.save(model_without_wrapper.state_dict(), history_path)

            # 定期验证
            if (epoch + 1) % val_interval == 0:
                val_metrics = evaluate(
                    model,
                    val_dataloader,
                    use_amp,
                    amp_dtype,
                    query_score_threshold,
                    instance_mask_threshold,
                    instance_match_iou_threshold,
                    instance_min_area,
                    instance_top_k,
                    instance_use_cls_score,
                )
                logger.info(
                    f'Validation Epoch: {epoch}, '
                    f'PP_InstF1: {val_metrics["pp_inst_f1"]:.4f}, '
                    f'PP_InstIoU: {val_metrics["pp_inst_mean_iou"]:.4f}, '
                    f'PP_MatchedIoU: {val_metrics["pp_inst_matched_iou"]:.4f}, '
                    f'PP_InstP: {val_metrics["pp_inst_precision"]:.4f}, '
                    f'PP_InstR: {val_metrics["pp_inst_recall"]:.4f}, '
                    f'PP_Pred: {val_metrics["pp_inst_pred_count"]:.1f}, '
                    f'PP_GT: {val_metrics["pp_inst_gt_count"]:.1f}, '
                    f'AlignedInstF1: {val_metrics["aligned_inst_f1"]:.4f}, '
                    f'AlignedInstIoU: {val_metrics["aligned_inst_mean_iou"]:.4f}, '
                    f'AlignedMatchedIoU: {val_metrics["aligned_inst_matched_iou"]:.4f}'
                )
                # 按 PP_InstF1 保存最优模型
                if val_metrics['pp_inst_f1'] > best_inst_f1:
                    best_inst_f1 = val_metrics['pp_inst_f1']
                    best_inst_f1_path = os.path.join(save_root, 'best_inst_f1.pth')
                    best_inst_f1_checkpoint = dict(checkpoint)
                    best_inst_f1_checkpoint['val_metrics'] = val_metrics
                    torch.save(best_inst_f1_checkpoint, best_inst_f1_path)
                    logger.info(f'>>> Best InstF1 updated to {best_inst_f1:.4f}. Model saved to {best_inst_f1_path}')
                model.train()

            # 学习率调度
            if scheduler is not None:
                scheduler.step()

    except Exception as e:
        # 异常中断时紧急保存权重
        emergency_path = os.path.join(save_root, 'emergency_exit_model.pth')
        torch.save(model_without_wrapper.state_dict(), emergency_path)
        logger.error(f'!!! 训练因错误中断: {str(e)}')
        logger.error(f'!!! 紧急权重已保存至: {emergency_path}')
        raise e

    logger.info("Training Finished!")

if __name__ == '__main__':
    main()
