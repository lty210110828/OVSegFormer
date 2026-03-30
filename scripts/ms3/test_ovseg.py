import sys
import os
import csv

# 将项目根目录加入 python 搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
import time
import numpy as np
import argparse
import cv2
from mmcv import Config
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import pyutils
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset
from scripts.ms3.utility import (
    build_instance_label_map_from_masks,
    collect_postprocessed_instance_diagnostics,
    compute_aligned_instance_metrics,
    compute_postprocessed_instance_metrics,
    create_instance_diagnostic_panel,
    instances_to_label_map,
    postprocess_instance_predictions,
)

def parse_thresholds(args, cfg):
    if args.thresholds:
        thresholds = [float(x.strip()) for x in args.thresholds.split(',') if x.strip()]
    elif args.threshold is not None:
        thresholds = [float(args.threshold)]
    else:
        thresholds = [float(getattr(cfg, 'metric_threshold', 0.5))]
    if len(thresholds) == 0:
        raise ValueError('至少需要提供一个阈值')
    thresholds = [min(1.0, max(0.0, x)) for x in thresholds]
    return thresholds

def compute_basic_metrics(pred_logits, target, threshold=0.5, eps=1e-6):
    """ 计算基础评估指标 (IoU, Dice, Precision, Recall, F1) """
    prob = torch.sigmoid(pred_logits)
    pred = (prob > threshold).float()
    target = target.float()
    
    # 确保维度匹配 [B, 1, H, W]
    if pred.dim() == 3: pred = pred.unsqueeze(1)
    if target.dim() == 3: target = target.unsqueeze(1)
    
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
    """ 计算组学覆盖率 (Omics Coverage) """
    prob = torch.sigmoid(pred_logits)
    pred = (prob > threshold).float()
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    h, w = mask_size
    coverages = []
    for b, pts in enumerate(omics_x):
        if pts.numel() == 0:
            continue
        # 将归一化坐标 [0, 1] 映射回像素坐标
        x = (pts[:, 0] * (w - 1)).long().clamp(0, w - 1)
        y = (pts[:, 1] * (h - 1)).long().clamp(0, h - 1)
        inside = pred[b, y, x]
        coverages.append(inside.float().mean().item())
    if len(coverages) == 0:
        return 0.0
    return float(sum(coverages) / len(coverages))

def aggregate_query_masks(output_cls, output_mask, query_valid_mask, query_score_threshold=None):
    query_scores = torch.sigmoid(output_cls.squeeze(-1))
    query_scores = query_scores.masked_fill(~query_valid_mask, 0.0)
    keep_mask = query_valid_mask
    if query_score_threshold is not None:
        keep_mask = query_valid_mask & (query_scores >= query_score_threshold)
        any_valid = query_valid_mask.any(dim=1)
        no_keep = ~keep_mask.any(dim=1) & any_valid
        if no_keep.any():
            masked_scores = query_scores.masked_fill(~query_valid_mask, -1.0)
            top_idx = masked_scores.argmax(dim=1)
            keep_mask[no_keep, :] = False
            keep_mask[no_keep, top_idx[no_keep]] = True
    mask_prob = torch.sigmoid(output_mask)
    weighted_prob = mask_prob * query_scores[:, :, None, None]
    weighted_prob = weighted_prob.masked_fill(~keep_mask[:, :, None, None], 0.0)
    pred_prob = weighted_prob.max(dim=1, keepdim=True)[0]
    pred_prob = pred_prob.clamp_(1e-4, 1.0 - 1e-4)
    return torch.logit(pred_prob)

def collect_query_diagnostics(output_cls, query_valid_mask, query_score_threshold):
    query_scores = torch.sigmoid(output_cls.squeeze(-1))
    query_scores = query_scores.masked_fill(~query_valid_mask, 0.0)
    keep_mask = query_valid_mask & (query_scores >= query_score_threshold)
    any_valid = query_valid_mask.any(dim=1)
    no_keep = ~keep_mask.any(dim=1) & any_valid
    if no_keep.any():
        masked_scores = query_scores.masked_fill(~query_valid_mask, -1.0)
        top_idx = masked_scores.argmax(dim=1)
        keep_mask[no_keep, :] = False
        keep_mask[no_keep, top_idx[no_keep]] = True

    batch_rows = []
    for b in range(query_scores.shape[0]):
        valid_scores = query_scores[b][query_valid_mask[b]]
        kept_scores = query_scores[b][keep_mask[b]]
        if valid_scores.numel() == 0:
            row = {
                'valid_queries': 0,
                'kept_queries': 0,
                'keep_ratio': 0.0,
                'score_mean': 0.0,
                'score_max': 0.0,
                'score_p50': 0.0,
                'score_p90': 0.0,
                'kept_mean': 0.0,
            }
        else:
            row = {
                'valid_queries': int(valid_scores.numel()),
                'kept_queries': int(kept_scores.numel()),
                'keep_ratio': float(kept_scores.numel() / valid_scores.numel()),
                'score_mean': float(valid_scores.mean().item()),
                'score_max': float(valid_scores.max().item()),
                'score_p50': float(valid_scores.quantile(0.5).item()),
                'score_p90': float(valid_scores.quantile(0.9).item()),
                'kept_mean': float(kept_scores.mean().item()) if kept_scores.numel() > 0 else 0.0,
            }
        batch_rows.append(row)
    return batch_rows

def custom_collate_fn(batch):
    """ 自定义整理函数，处理变长的点云数据和质心 """
    imgs = torch.stack([item['img'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    omics_x = [item['omics_x'] for item in batch]
    omics_gene_ids = [item['omics_gene_ids'] for item in batch]
    omics_qv = [item['omics_qv'] for item in batch]
    centroids = [item['centroids'] for item in batch]
    instance_masks = [item['instance_masks'] for item in batch]
    instance_target_valid = [item['instance_target_valid'] for item in batch]
    
    max_cells = max([c.shape[0] for c in centroids])
    if max_cells == 0: max_cells = 1 
    
    padded_centroids = []
    centroid_valid_mask = []
    padded_instance_masks = []
    padded_instance_target_valid = []
    for c in centroids:
        n = c.shape[0]
        idx = len(padded_centroids)
        cur_instance_masks = instance_masks[idx]
        cur_instance_valid = instance_target_valid[idx]
        if n < max_cells:
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
    
    padded_centroids = torch.stack(padded_centroids)
    centroid_valid_mask = torch.stack(centroid_valid_mask)
    padded_instance_masks = torch.stack(padded_instance_masks)
    padded_instance_target_valid = torch.stack(padded_instance_target_valid)
    query_valid_mask = centroid_valid_mask & padded_instance_target_valid
    return {
        'img': imgs,
        'mask': masks,
        'omics_x': omics_x,
        'omics_gene_ids': omics_gene_ids,
        'omics_qv': omics_qv,
        'centroids': padded_centroids,
        'instance_masks': padded_instance_masks,
        'instance_target_valid': padded_instance_target_valid,
        'centroid_valid_mask': centroid_valid_mask,
        'query_valid_mask': query_valid_mask
    }

def main():
    parser = argparse.ArgumentParser(description='OVSegFormer 测试脚本')
    parser.add_argument('cfg', type=str, help='配置文件路径')
    parser.add_argument('weights', type=str, help='权重文件路径 (.pth)')
    parser.add_argument('--save_dir', type=str, default='eval_results', help='结果保存目录')
    parser.add_argument('--save_mask', action='store_true', help='是否保存预测生成的 Mask 图像')
    parser.add_argument('--device', type=str, default='cuda', help='测试设备')
    parser.add_argument('--threshold', type=float, default=None, help='单个二值化阈值，覆盖配置中的 metric_threshold')
    parser.add_argument('--thresholds', type=str, default='', help='逗号分隔的多阈值列表，例如 0.5,0.6,0.7,0.8')
    parser.add_argument('--diagnose_queries', action='store_true', help='输出 query 分数与保留数量诊断信息')
    parser.add_argument('--save_instances', action='store_true', help='是否保存实例标签图和实例元信息')
    parser.add_argument('--save_diagnostics', action='store_true', help='是否保存实例诊断面板和 query 掩膜')
    args = parser.parse_args()

    # 1. 加载配置
    cfg = Config.fromfile(args.cfg)
    threshold_values = parse_thresholds(args, cfg)
    query_score_threshold = getattr(cfg, 'query_score_threshold', 0.5)
    instance_mask_threshold = getattr(cfg, 'instance_mask_threshold', threshold_values[0])
    instance_match_iou_threshold = getattr(cfg, 'instance_match_iou_threshold', 0.5)
    instance_overlap_threshold = getattr(cfg, 'instance_overlap_threshold', 0.5)
    instance_min_area = getattr(cfg, 'instance_min_area', 16)
    instance_top_k = getattr(cfg, 'instance_top_k', None)
    instance_use_cls_score = bool(getattr(cfg, 'instance_use_cls_score', False))
    diagnostic_max_cases = getattr(cfg, 'diagnostic_max_cases', 50)
    diagnostic_query_topk = getattr(cfg, 'diagnostic_query_topk', 5)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    logger = getLogger(os.path.join(args.save_dir, 'eval.log'), __name__)
    logger.info(f'Config loaded from {args.cfg}')
    logger.info(f'Weights: {args.weights}')
    logger.info(f'Thresholds: {", ".join(f"{x:.2f}" for x in threshold_values)}')
    logger.info(f'Query score threshold: {query_score_threshold:.2f}')

    # 2. 构建模型并加载权重
    model = build_model(**cfg.model)
    
    # 兼容两种权重格式: latest.pth (含 dict) 和 epoch_xx.pth (纯 state_dict)
    checkpoint = torch.load(args.weights, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        logger.info(f"Detected checkpoint format with metadata (Epoch: {checkpoint.get('epoch', 'N/A')})")
    else:
        state_dict = checkpoint
        logger.info("Detected pure state_dict format")
    
    # 处理 DataParallel 保存时可能产生的 .module 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(args.device)
    model.eval()

    # 3. 构建测试数据集
    # 优先使用 cfg.dataset.test，如果不存在则回退到 cfg.dataset.val
    test_cfg = getattr(cfg.dataset, 'test', cfg.dataset.val)
    test_dataset = build_dataset(**test_cfg)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=getattr(test_cfg, 'batch_size', 1),
        shuffle=False,
        num_workers=cfg.process.num_works,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # 4. 推理循环
    instance_metrics_by_threshold = {thr: [] for thr in threshold_values}
    aligned_instance_metrics = []
    query_diagnostics = []
    instance_rows = []
    diagnostic_sample_rows = []
    diagnostic_instance_rows = []
    saved_diagnostic_cases = 0
    
    if args.save_mask:
        if len(threshold_values) == 1:
            mask_dirs = {threshold_values[0]: os.path.join(args.save_dir, 'pred_masks')}
        else:
            mask_dirs = {
                thr: os.path.join(args.save_dir, 'pred_masks', f"thr_{thr:.2f}".replace('.', '_'))
                for thr in threshold_values
            }
        for mask_dir in mask_dirs.values():
            os.makedirs(mask_dir, exist_ok=True)
    if args.save_instances:
        instance_dir = os.path.join(args.save_dir, 'instance_maps')
        os.makedirs(instance_dir, exist_ok=True)
    if args.save_diagnostics:
        diagnostic_dir = os.path.join(args.save_dir, 'diagnostics')
        diagnostic_panel_dir = os.path.join(diagnostic_dir, 'panels')
        diagnostic_query_dir = os.path.join(diagnostic_dir, 'query_masks')
        os.makedirs(diagnostic_panel_dir, exist_ok=True)
        os.makedirs(diagnostic_query_dir, exist_ok=True)

    logger.info("Starting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="Eval")):
            imgs = batch['img'].to(args.device)
            masks = batch['mask'].to(args.device)
            centroids = batch['centroids'].to(args.device)
            query_valid_mask = batch['query_valid_mask'].to(args.device)
            instance_masks = batch['instance_masks'].to(args.device)
            instance_target_valid = batch['instance_target_valid'].to(args.device)
            omics_x = [x.to(args.device) for x in batch['omics_x']]
            omics_gene_ids = [x.to(args.device) for x in batch['omics_gene_ids']]
            omics_qv = [x.to(args.device) for x in batch['omics_qv']]

            # 前向传播
            # 模型输出: (outputs_class, outputs_mask)
            # outputs_class: [B, N, 1]
            # outputs_mask: [B, N, H, W]
            output_cls, output_mask = model(imgs, omics_x, centroids, omics_gene_ids=omics_gene_ids, omics_qv=omics_qv)
            if args.diagnose_queries:
                batch_rows = collect_query_diagnostics(output_cls, query_valid_mask, query_score_threshold)
                for offset, row in enumerate(batch_rows):
                    row['sample_index'] = i * imgs.shape[0] + offset
                    query_diagnostics.append(row)
            post_instances = postprocess_instance_predictions(
                output_cls.detach(),
                output_mask.detach(),
                query_valid_mask,
                target_size=masks.shape[-2:],
                query_score_threshold=query_score_threshold if instance_use_cls_score else None,
                mask_threshold=instance_mask_threshold,
                overlap_threshold=instance_overlap_threshold,
                min_area=instance_min_area,
                top_k=instance_top_k,
                use_cls_score=instance_use_cls_score,
            )
            aligned_metrics = compute_aligned_instance_metrics(
                output_cls.detach(),
                output_mask.detach(),
                instance_masks,
                instance_target_valid,
                query_score_threshold=query_score_threshold if instance_use_cls_score else None,
                mask_threshold=instance_mask_threshold,
                match_iou_threshold=instance_match_iou_threshold,
            )
            aligned_instance_metrics.append(aligned_metrics)
            sample_diag_rows, sample_instance_diag_rows = collect_postprocessed_instance_diagnostics(
                post_instances,
                instance_masks,
                instance_target_valid,
                match_iou_threshold=instance_match_iou_threshold,
            )
            if args.save_instances:
                for batch_offset, sample_instances in enumerate(post_instances):
                    sample_index = i * imgs.shape[0] + batch_offset
                    label_map = instances_to_label_map(sample_instances, masks.shape[-2:])
                    cv2.imwrite(
                        os.path.join(instance_dir, f'inst_{sample_index:04d}.png'),
                        label_map.astype(np.uint16)
                    )
                    if len(sample_instances) == 0:
                        instance_rows.append({
                            'sample_index': sample_index,
                            'instance_id': -1,
                            'query_idx': -1,
                            'score': 0.0,
                            'area': 0,
                        })
                    else:
                        for inst_id, inst in enumerate(sample_instances, start=1):
                            instance_rows.append({
                                'sample_index': sample_index,
                                'instance_id': inst_id,
                                'query_idx': inst['query_idx'],
                                'score': inst['score'],
                                'area': inst['area'],
                            })
            for batch_offset, row in enumerate(sample_diag_rows):
                sample_index = i * imgs.shape[0] + batch_offset
                row['sample_index'] = sample_index
                diagnostic_sample_rows.append(row)
                for inst_row in sample_instance_diag_rows[batch_offset]:
                    cur_row = dict(inst_row)
                    cur_row['sample_index'] = sample_index
                    diagnostic_instance_rows.append(cur_row)
                if args.save_diagnostics and saved_diagnostic_cases < int(diagnostic_max_cases):
                    gt_valid = instance_target_valid[batch_offset].bool()
                    gt_label_map = build_instance_label_map_from_masks(instance_masks[batch_offset][gt_valid])
                    pred_label_map = instances_to_label_map(post_instances[batch_offset], masks.shape[-2:])
                    panel = create_instance_diagnostic_panel(imgs[batch_offset], gt_label_map, pred_label_map)
                    cv2.imwrite(
                        os.path.join(diagnostic_panel_dir, f'diag_{sample_index:04d}.png'),
                        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                    )
                    sample_query_dir = os.path.join(diagnostic_query_dir, f'sample_{sample_index:04d}')
                    os.makedirs(sample_query_dir, exist_ok=True)
                    for inst in post_instances[batch_offset][:int(diagnostic_query_topk)]:
                        query_mask = (inst['mask'].numpy().astype(np.uint8) * 255)
                        file_name = f"query_{inst['query_idx']:03d}_score_{inst['score']:.3f}.png"
                        cv2.imwrite(os.path.join(sample_query_dir, file_name), query_mask)
                    saved_diagnostic_cases += 1
            
            for threshold in threshold_values:
                cur_instances = postprocess_instance_predictions(
                    output_cls.detach(),
                    output_mask.detach(),
                    query_valid_mask,
                    target_size=masks.shape[-2:],
                    query_score_threshold=query_score_threshold if instance_use_cls_score else None,
                    mask_threshold=threshold,
                    overlap_threshold=instance_overlap_threshold,
                    min_area=instance_min_area,
                    top_k=instance_top_k,
                    use_cls_score=instance_use_cls_score,
                )
                instance_metrics = compute_postprocessed_instance_metrics(
                    cur_instances,
                    instance_masks,
                    instance_target_valid,
                    match_iou_threshold=instance_match_iou_threshold,
                )
                instance_metrics_by_threshold[threshold].append(instance_metrics)

                if args.save_mask:
                    for batch_offset, sample_instances in enumerate(cur_instances):
                        sample_index = i * imgs.shape[0] + batch_offset
                        pred_np = (instances_to_label_map(sample_instances, masks.shape[-2:]) > 0).astype(np.uint8) * 255
                        cv2.imwrite(os.path.join(mask_dirs[threshold], f'pred_{sample_index:04d}.png'), pred_np)

    summary_rows = []
    logger.info("-" * 30)
    for threshold in threshold_values:
        final_instance_metrics = {}
        for key in instance_metrics_by_threshold[threshold][0].keys():
            final_instance_metrics[key] = np.mean([m[key] for m in instance_metrics_by_threshold[threshold]])
        summary_rows.append((
            threshold,
            final_instance_metrics['pp_inst_f1'],
            final_instance_metrics['pp_inst_mean_iou'],
            final_instance_metrics['pp_inst_matched_iou'],
            final_instance_metrics['pp_inst_precision'],
            final_instance_metrics['pp_inst_recall'],
        ))
        logger.info(f"Evaluation Final Results @ instance_threshold={threshold:.2f}:")
        for k, v in final_instance_metrics.items():
            logger.info(f"{k:10s}: {v:.4f}")
        logger.info("-" * 30)
    if len(aligned_instance_metrics) > 0:
        logger.info("Aligned Query Metrics:")
        aligned_summary = {
            'inst_f1': np.mean([m['inst_f1'] for m in aligned_instance_metrics]),
            'inst_mean_iou': np.mean([m['inst_mean_iou'] for m in aligned_instance_metrics]),
            'inst_matched_iou': np.mean([m['inst_matched_iou'] for m in aligned_instance_metrics]),
        }
        for k, v in aligned_summary.items():
            logger.info(f"{k:18s}: {v:.4f}")
        logger.info("-" * 30)
    if len(summary_rows) > 1:
        logger.info("Instance Threshold Sweep Summary:")
        for threshold, inst_f1, inst_iou, inst_matched_iou, inst_precision, inst_recall in summary_rows:
            logger.info(
                f"thr={threshold:.2f}, InstF1={inst_f1:.4f}, InstIoU={inst_iou:.4f}, "
                f'InstMatchIoU={inst_matched_iou:.4f}, InstP={inst_precision:.4f}, InstR={inst_recall:.4f}'
            )
    if args.diagnose_queries and len(query_diagnostics) > 0:
        logger.info("Query Diagnostics Summary:")
        keys = ['valid_queries', 'kept_queries', 'keep_ratio', 'score_mean', 'score_max', 'score_p50', 'score_p90', 'kept_mean']
        for key in keys:
            values = np.array([row[key] for row in query_diagnostics], dtype=np.float64)
            logger.info(
                f"{key}: mean={values.mean():.4f}, min={values.min():.4f}, "
                f"p50={np.percentile(values, 50):.4f}, p90={np.percentile(values, 90):.4f}, max={values.max():.4f}"
            )
        diag_path = os.path.join(args.save_dir, 'query_diagnostics.csv')
        with open(diag_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_index'] + keys)
            writer.writeheader()
            writer.writerows(query_diagnostics)
        logger.info(f"Query diagnostics saved to {diag_path}")
    if args.save_diagnostics:
        sample_diag_path = os.path.join(args.save_dir, 'diagnostic_samples.csv')
        with open(sample_diag_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['sample_index', 'gt_count', 'pred_count', 'matched_count', 'fp_count', 'fn_count', 'mean_matched_iou', 'mean_pred_score']
            )
            writer.writeheader()
            writer.writerows(diagnostic_sample_rows)
        instance_diag_path = os.path.join(args.save_dir, 'diagnostic_instances.csv')
        with open(instance_diag_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['sample_index', 'instance_id', 'query_idx', 'score', 'area', 'matched_gt_id', 'matched_iou', 'status']
            )
            writer.writeheader()
            writer.writerows(diagnostic_instance_rows)
        logger.info(f"Diagnostic sample summary saved to {sample_diag_path}")
        logger.info(f"Diagnostic instance summary saved to {instance_diag_path}")
    if args.save_instances:
        meta_path = os.path.join(args.save_dir, 'instance_predictions.csv')
        with open(meta_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_index', 'instance_id', 'query_idx', 'score', 'area'])
            writer.writeheader()
            writer.writerows(instance_rows)
        logger.info(f"Instance prediction metadata saved to {meta_path}")
    logger.info(f"Results saved to {args.save_dir}")

if __name__ == '__main__':
    main()
