import torch
from torch.nn import functional as F

import os
import shutil
import logging
import cv2
import numpy as np
from PIL import Image

import sys
import time
import pandas as pd
import pdb
from torchvision import transforms
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ============================================================
# [原 AVSegFormer 代码 — 保留但不注释]
# ============================================================

def save_checkpoint(state, epoch, is_best, checkpoint_dir='./models', filename='checkpoint', thres=100):
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if epoch >= thres:
        file_path = os.path.join(checkpoint_dir, filename + '_{}'.format(str(epoch)) + '.pth.tar')
    else:
        file_path = os.path.join(checkpoint_dir, filename + '.pth.tar')
    torch.save(state, file_path)
    logger.info('==> save model at {}'.format(file_path))
    if is_best:
        cpy_file = os.path.join(checkpoint_dir, filename + '_model_best.pth.tar')
        shutil.copyfile(file_path, cpy_file)
        logger.info('==> save best model at {}'.format(cpy_file))


# ============================================================
# [OVSegFormer 实例评估函数]
# ============================================================

def compute_aligned_instance_metrics(
    output_cls,
    output_mask,
    instance_masks,
    instance_target_valid,
    query_score_threshold=0.5,
    mask_threshold=0.5,
    match_iou_threshold=0.5,
    force_keep_top1=True,
    eps=1e-6,
):
    """
    Aligned 实例指标计算 — 直接用 Hungarian 匹配 query 到 GT (跳过后处理)

    流程:
        1. 将 output_mask 插值到 GT 尺寸
        2. 按 query_score 过滤有效预测
        3. 二值化后计算 pred↔gt 的 IoU 矩阵
        4. 用 linear_sum_assignment 做 Hungarian 最优匹配
        5. 按 match_iou_threshold 判定 TP/FP/FN → 计算 Precision/Recall/F1

    Returns:
        dict: inst_precision, inst_recall, inst_f1, inst_mean_iou, inst_matched_iou, ...
    """
    if output_mask.shape[-2:] != instance_masks.shape[-2:]:
        flat_masks = output_mask.reshape(-1, 1, output_mask.shape[-2], output_mask.shape[-1])
        flat_masks = F.interpolate(flat_masks, size=instance_masks.shape[-2:], mode='bilinear', align_corners=False)
        output_mask = flat_masks.reshape(instance_masks.shape[0], instance_masks.shape[1], instance_masks.shape[-2], instance_masks.shape[-1])

    query_scores = torch.sigmoid(output_cls.squeeze(-1))
    valid_mask = instance_target_valid.bool()
    keep_mask = valid_mask if query_score_threshold is None else (valid_mask & (query_scores >= query_score_threshold))
    # 兜底: 如果某样本所有 query 都低于阈值，保留最高分的那个
    if query_score_threshold is not None:
        if force_keep_top1:
            any_valid = valid_mask.any(dim=1)
            no_keep = ~keep_mask.any(dim=1) & any_valid
            if no_keep.any():
                masked_scores = query_scores.masked_fill(~valid_mask, -1.0)
                top_idx = masked_scores.argmax(dim=1)
                keep_mask[no_keep] = False
                keep_mask[no_keep, top_idx[no_keep]] = True

    pred_masks = (torch.sigmoid(output_mask) > mask_threshold).float()
    target_masks = (instance_masks > 0.5).float()

    total_gt = 0.0
    total_pred = 0.0
    total_match = 0.0
    total_iou = 0.0
    total_overlap = 0.0
    total_samples = 0.0
    matched_iou_values = []

    for batch_idx in range(pred_masks.shape[0]):
        cur_gt_valid = valid_mask[batch_idx]
        cur_target = target_masks[batch_idx][cur_gt_valid]
        cur_pred_keep = keep_mask[batch_idx]
        cur_pred = pred_masks[batch_idx][cur_pred_keep]

        total_gt += float(cur_target.shape[0])
        total_pred += float(cur_pred.shape[0])
        if cur_target.shape[0] == 0 or cur_pred.shape[0] == 0:
            continue

        # 计算 IoU 矩阵 + Hungarian 匹配
        pred_flat = cur_pred.flatten(1)
        target_flat = cur_target.flatten(1)
        inter = pred_flat @ target_flat.t()
        pred_area = pred_flat.sum(-1, keepdim=True)
        target_area = target_flat.sum(-1).unsqueeze(0)
        union = pred_area + target_area - inter
        iou_matrix = inter / (union + eps)
        pred_indices, gt_indices = linear_sum_assignment((1.0 - iou_matrix).detach().cpu().numpy())
        matched_ious = iou_matrix[pred_indices, gt_indices]
        total_iou += float(matched_ious.sum().item())
        total_samples += float(matched_ious.numel())

        matched = matched_ious >= match_iou_threshold
        total_match += float(matched.sum().item())
        if matched.any():
            matched_iou_values.append(matched_ious[matched])

        # 计算预测实例间的平均重叠率 (Overlap metric)
        kept_pred = cur_pred
        if kept_pred.shape[0] > 1:
            kept_flat = kept_pred.flatten(1)
            kept_area = kept_flat.sum(-1)
            pair_inter = kept_flat @ kept_flat.t()
            denom = torch.minimum(kept_area[:, None], kept_area[None, :]).clamp_min(eps)
            overlap = pair_inter / denom
            pair_mask = torch.triu(torch.ones_like(overlap, dtype=torch.bool), diagonal=1)
            if pair_mask.any():
                total_overlap += float(overlap[pair_mask].mean().item())

    inst_precision = total_match / max(total_pred, eps)
    inst_recall = total_match / max(total_gt, eps)
    inst_f1 = 2 * inst_precision * inst_recall / max(inst_precision + inst_recall, eps)
    inst_mean_iou = total_iou / max(total_samples, eps)
    inst_matched_iou = 0.0
    if len(matched_iou_values) > 0:
        inst_matched_iou = float(torch.cat(matched_iou_values).mean().item())
    inst_keep_ratio = total_pred / max(total_gt, eps)
    inst_overlap = total_overlap / max(float(pred_masks.shape[0]), 1.0)

    return {
        'inst_precision': float(inst_precision),
        'inst_recall': float(inst_recall),
        'inst_f1': float(inst_f1),
        'inst_mean_iou': float(inst_mean_iou),
        'inst_matched_iou': float(inst_matched_iou),
        'inst_keep_ratio': float(inst_keep_ratio),
        'inst_overlap': float(inst_overlap),
        'inst_pred_count': float(total_pred),
        'inst_gt_count': float(total_gt),
        'inst_match_count': float(total_match),
    }


def postprocess_instance_predictions(
    output_cls,
    output_mask,
    query_valid_mask,
    target_size=None,
    query_score_threshold=None,
    mask_threshold=0.5,
    overlap_threshold=0.5,
    min_area=16,
    top_k=None,
    force_keep_top1=True,
    use_cls_score=False,
    eps=1e-6,
):
    """
    后处理实例预测 — 将 query 输出转换为离散实例掩膜列表

    流程:
        1. 按 query_score 过滤低分 query (可选)
        2. 二值化掩膜
        3. 像素归属 NMS: 每个像素只属于 ownership_prob 最高的 query
        4. 过滤面积 < min_area 的实例
        5. 可选: 按 score 取 top-K

    Args:
        output_cls: [B, N, 1] objectness logits
        output_mask: [B, N, H, W] 掩膜 logits
        query_valid_mask: [B, N] 有效 query 布尔掩膜

    Returns:
        batch_instances: list of list of dict, 每个 dict 包含:
            query_idx, score, area, mask (二值 tensor)
    """
    if target_size is not None and output_mask.shape[-2:] != tuple(target_size):
        flat_masks = output_mask.reshape(-1, 1, output_mask.shape[-2], output_mask.shape[-1])
        flat_masks = F.interpolate(flat_masks, size=target_size, mode='bilinear', align_corners=False)
        output_mask = flat_masks.reshape(output_mask.shape[0], output_mask.shape[1], target_size[0], target_size[1])

    query_scores = torch.sigmoid(output_cls.squeeze(-1))
    mask_probs = torch.sigmoid(output_mask)
    batch_instances = []

    for batch_idx in range(output_cls.shape[0]):
        valid_idx = torch.nonzero(query_valid_mask[batch_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            batch_instances.append([])
            continue

        # 按 query 分数过滤
        cur_scores = query_scores[batch_idx, valid_idx]
        keep = torch.ones_like(cur_scores, dtype=torch.bool)
        if use_cls_score and query_score_threshold is not None:
            keep = cur_scores >= query_score_threshold
            if keep.sum() == 0 and force_keep_top1:
                top_local_idx = cur_scores.argmax()
                keep = torch.zeros_like(cur_scores, dtype=torch.bool)
                keep[top_local_idx] = True
        kept_idx = valid_idx[keep]
        kept_scores = cur_scores[keep]
        kept_masks = mask_probs[batch_idx, kept_idx]
        if kept_idx.numel() == 0:
            batch_instances.append([])
            continue

        # 像素归属 NMS: 每个像素只属于概率最高的 query
        ownership_prob = kept_masks
        if use_cls_score:
            ownership_prob = ownership_prob * kept_scores[:, None, None]
        best_prob, best_local_idx = ownership_prob.max(dim=0)
        foreground = best_prob > mask_threshold

        selected_instances = []
        for local_rank, query_idx in enumerate(kept_idx.tolist()):
            bin_mask = foreground & (best_local_idx == local_rank)
            area = int(bin_mask.sum().item())
            if area < int(min_area):  # 过滤过小的实例
                continue
            mask_values = kept_masks[local_rank][bin_mask]
            score = float(mask_values.mean().item()) if mask_values.numel() > 0 else float(kept_scores[local_rank].item())
            selected_instances.append({
                'query_idx': int(query_idx),
                'score': score,
                'area': area,
                'mask': bin_mask.detach().cpu(),
            })
        if top_k is not None and len(selected_instances) > int(top_k):
            selected_instances = sorted(selected_instances, key=lambda x: x['score'], reverse=True)[:int(top_k)]
        batch_instances.append(selected_instances)

    return batch_instances


def instances_to_label_map(instances, image_size):
    """将实例列表转换为标签图 (每个实例分配唯一整数 ID)"""
    label_map = np.zeros(image_size, dtype=np.int32)
    for inst_id, inst in enumerate(instances, start=1):
        mask = inst['mask'].numpy().astype(bool)
        label_map[mask] = inst_id
    return label_map


def compute_postprocessed_instance_metrics(
    batch_instances,
    instance_masks,
    instance_target_valid,
    match_iou_threshold=0.5,
    eps=1e-6,
):
    """
    后处理实例指标计算 — 评估 postprocess_instance_predictions 的输出

    与 compute_aligned_instance_metrics 的区别:
        - 输入是后处理后的实例列表 (而非原始 query 输出)
        - 更接近实际部署时的评估方式
    """
    total_gt = 0.0
    total_pred = 0.0
    total_match = 0.0
    total_iou = 0.0
    total_pairs = 0.0
    total_overlap = 0.0
    matched_iou_values = []

    for batch_idx, instances in enumerate(batch_instances):
        gt_masks = instance_masks[batch_idx][instance_target_valid[batch_idx].bool()].float()
        pred_masks = []
        for inst in instances:
            pred_masks.append(inst['mask'].to(device=gt_masks.device, dtype=torch.float32))
        total_gt += float(gt_masks.shape[0])
        total_pred += float(len(pred_masks))
        if gt_masks.shape[0] == 0 or len(pred_masks) == 0:
            continue

        pred_masks = torch.stack(pred_masks, dim=0)
        # IoU 矩阵 + Hungarian 匹配
        pred_flat = pred_masks.flatten(1)
        gt_flat = gt_masks.flatten(1)
        inter = pred_flat @ gt_flat.t()
        pred_area = pred_flat.sum(-1, keepdim=True)
        gt_area = gt_flat.sum(-1).unsqueeze(0)
        union = pred_area + gt_area - inter
        iou_matrix = inter / (union + eps)
        pred_idx, gt_idx = linear_sum_assignment((1.0 - iou_matrix).cpu().numpy())
        matched_ious = iou_matrix[pred_idx, gt_idx]
        total_iou += float(matched_ious.sum().item())
        total_pairs += float(matched_ious.numel())
        matched = matched_ious >= match_iou_threshold
        total_match += float(matched.sum().item())
        if matched.any():
            matched_iou_values.append(matched_ious[matched])

        # 预测实例间重叠率
        if pred_masks.shape[0] > 1:
            pair_inter = pred_flat @ pred_flat.t()
            denom = torch.minimum(pred_area, pred_area.t()).clamp_min(eps)
            overlap = pair_inter / denom
            pair_mask = torch.triu(torch.ones_like(overlap, dtype=torch.bool), diagonal=1)
            if pair_mask.any():
                total_overlap += float(overlap[pair_mask].mean().item())

    precision = total_match / max(total_pred, eps)
    recall = total_match / max(total_gt, eps)
    f1 = 2 * precision * recall / max(precision + recall, eps)
    mean_iou = total_iou / max(total_pairs, eps)
    matched_iou = 0.0
    if len(matched_iou_values) > 0:
        matched_iou = float(torch.cat(matched_iou_values).mean().item())
    return {
        'pp_inst_precision': float(precision),
        'pp_inst_recall': float(recall),
        'pp_inst_f1': float(f1),
        'pp_inst_mean_iou': float(mean_iou),
        'pp_inst_matched_iou': float(matched_iou),
        'pp_inst_overlap': float(total_overlap / max(float(len(batch_instances)), 1.0)),
        'pp_inst_pred_count': float(total_pred),
        'pp_inst_gt_count': float(total_gt),
        'pp_inst_match_count': float(total_match),
    }


def build_instance_label_map_from_masks(masks):
    """将一组二值掩膜转换为标签图 (每个掩膜分配唯一 ID)"""
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    label_map = np.zeros(masks.shape[-2:], dtype=np.int32)
    for inst_id, mask in enumerate(masks, start=1):
        label_map[mask > 0.5] = inst_id
    return label_map


def colorize_label_map(label_map):
    """将整数标签图转换为彩色可视化图 (每个实例一种颜色)"""
    label_map = np.asarray(label_map, dtype=np.int32)
    color = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    unique_ids = np.unique(label_map)
    for inst_id in unique_ids:
        if inst_id <= 0:
            continue
        color_value = np.array([
            (inst_id * 37) % 255,
            (inst_id * 97) % 255,
            (inst_id * 57) % 255,
        ], dtype=np.uint8)
        color[label_map == inst_id] = color_value
    return color


def tensor_to_uint8_image(img_tensor):
    """将 PyTorch tensor 转换为 uint8 numpy 图像 (用于可视化)"""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().float().numpy()
    else:
        img = np.asarray(img_tensor, dtype=np.float32)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    img_min = float(img.min())
    img_max = float(img.max())
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def create_instance_diagnostic_panel(img_tensor, gt_label_map, pred_label_map):
    """
    创建 2×2 诊断面板: 原图 | GT 标签图 / Pred 标签图 | Error Map

    Error Map 颜色编码:
        绿色 = TP (GT 和 Pred 都有)
        红色 = FN (GT 有但 Pred 没有)
        蓝色 = FP (Pred 有但 GT 没有)
    """
    img = tensor_to_uint8_image(img_tensor)
    gt_color = colorize_label_map(gt_label_map)
    pred_color = colorize_label_map(pred_label_map)
    gt_union = gt_label_map > 0
    pred_union = pred_label_map > 0
    error_map = np.zeros_like(img)
    error_map[np.logical_and(gt_union, pred_union)] = np.array([0, 255, 0], dtype=np.uint8)   # TP
    error_map[np.logical_and(gt_union, ~pred_union)] = np.array([255, 0, 0], dtype=np.uint8)  # FN
    error_map[np.logical_and(~gt_union, pred_union)] = np.array([0, 0, 255], dtype=np.uint8)  # FP
    top = np.concatenate([img, gt_color], axis=1)
    bottom = np.concatenate([pred_color, error_map], axis=1)
    return np.concatenate([top, bottom], axis=0)


def collect_postprocessed_instance_diagnostics(
    batch_instances,
    instance_masks,
    instance_target_valid,
    match_iou_threshold=0.5,
    eps=1e-6,
):
    """
    收集逐样本/逐实例的诊断数据

    Returns:
        sample_rows: 每个样本的汇总 (gt_count, pred_count, matched_count, ...)
        instance_rows: 每个实例的匹配详情 (query_idx, score, matched_gt_id, status=tp/fp)
    """
    sample_rows = []
    instance_rows = []

    for batch_idx, instances in enumerate(batch_instances):
        gt_masks = instance_masks[batch_idx][instance_target_valid[batch_idx].bool()].float()
        gt_count = int(gt_masks.shape[0])
        pred_count = int(len(instances))
        matched_count = 0
        mean_matched_iou = 0.0
        pred_to_gt = {}
        matched_iou_values = []

        if gt_count > 0 and pred_count > 0:
            # Hungarian 匹配
            pred_masks = torch.stack(
                [inst['mask'].to(device=gt_masks.device, dtype=torch.float32) for inst in instances],
                dim=0
            )
            pred_flat = pred_masks.flatten(1)
            gt_flat = gt_masks.flatten(1)
            inter = pred_flat @ gt_flat.t()
            pred_area = pred_flat.sum(-1, keepdim=True)
            gt_area = gt_flat.sum(-1).unsqueeze(0)
            union = pred_area + gt_area - inter
            iou_matrix = inter / (union + eps)
            pred_idx, gt_idx = linear_sum_assignment((1.0 - iou_matrix).cpu().numpy())
            for cur_pred_idx, cur_gt_idx in zip(pred_idx.tolist(), gt_idx.tolist()):
                cur_iou = float(iou_matrix[cur_pred_idx, cur_gt_idx].item())
                pred_to_gt[cur_pred_idx] = (cur_gt_idx, cur_iou)
                if cur_iou >= match_iou_threshold:
                    matched_count += 1
                    matched_iou_values.append(cur_iou)
            if len(matched_iou_values) > 0:
                mean_matched_iou = float(np.mean(matched_iou_values))

        fp_count = pred_count - matched_count
        fn_count = gt_count - matched_count
        mean_pred_score = float(np.mean([inst['score'] for inst in instances])) if pred_count > 0 else 0.0
        sample_rows.append({
            'gt_count': gt_count,
            'pred_count': pred_count,
            'matched_count': matched_count,
            'fp_count': fp_count,
            'fn_count': fn_count,
            'mean_matched_iou': mean_matched_iou,
            'mean_pred_score': mean_pred_score,
        })

        # 逐实例匹配状态 (tp/fp)
        sample_instance_rows = []
        for pred_idx, inst in enumerate(instances, start=1):
            matched_gt_id = -1
            matched_iou = 0.0
            status = 'fp'
            mapped = pred_to_gt.get(pred_idx - 1, None)
            if mapped is not None:
                matched_gt_id = mapped[0] + 1
                matched_iou = mapped[1]
                if matched_iou >= match_iou_threshold:
                    status = 'tp'
            sample_instance_rows.append({
                'instance_id': pred_idx,
                'query_idx': inst['query_idx'],
                'score': inst['score'],
                'area': inst['area'],
                'matched_gt_id': matched_gt_id,
                'matched_iou': matched_iou,
                'status': status,
            })
        instance_rows.append(sample_instance_rows)

    return sample_rows, instance_rows


# ============================================================
# [原 AVSegFormer 评估代码 — 保留但不注释]
# ============================================================

def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union + eps)) / N

    return iou


def _eval_pr(y_pred, y, num, cuda_flag=True):
    if cuda_flag:
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / \
            (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall


def Eval_Fmeasure(pred, gt, pr_num=255):
    r"""
        param:
            pred: size [N x H x W]
            gt: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    print('=> eval [FMeasure]..')
    pred = torch.sigmoid(pred)
    N = pred.size(0)
    beta2 = 0.3
    avg_f, img_num = 0.0, 0
    score = torch.zeros(pr_num)
    print("{} videos in this batch".format(N))

    for img_id in range(N):
        if torch.mean(gt[img_id]) == 0.0:
            continue
        prec, recall = _eval_pr(pred[img_id], gt[img_id], pr_num)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0
        avg_f += f_score
        img_num += 1
        score = avg_f / img_num

    return score.max().item()


def save_mask(pred_masks, save_base_path, video_name_list):
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)
    pred_masks = pred_masks.squeeze(2)
    pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()
    pred_masks = pred_masks.view(-1, 5,
                                 pred_masks.shape[-2], pred_masks.shape[-1])
    pred_masks = pred_masks.cpu().data.numpy().astype(np.uint8)
    pred_masks *= 255
    bs = pred_masks.shape[0]
    for idx in range(bs):
        video_name = video_name_list[idx]
        mask_save_path = os.path.join(save_base_path, video_name)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path, exist_ok=True)
        one_video_masks = pred_masks[idx]
        for video_id in range(len(one_video_masks)):
            one_mask = one_video_masks[video_id]
            output_name = "%s_%d.png" % (video_name, video_id)
            im = Image.fromarray(one_mask).convert('P')
            im.save(os.path.join(mask_save_path, output_name), format='PNG')


def save_raw_img_mask(anno_file_path, raw_img_base_path, mask_base_path, split='test', r=0.5):
    df = pd.read_csv(anno_file_path, sep=',')
    df_test = df[df['split'] == split]
    count = 0
    for video_id in range(len(df_test)):
        video_name = df_test.iloc[video_id][0]
        raw_img_path = os.path.join(raw_img_base_path, video_name)
        for img_id in range(5):
            img_name = "%s.mp4_%d.png" % (video_name, img_id + 1)
            raw_img = cv2.imread(os.path.join(raw_img_path, img_name))
            mask = cv2.imread(os.path.join(
                mask_base_path, 'pred_masks', video_name, "%s_%d.png" % (video_name, img_id)))
            raw_img_mask = cv2.addWeighted(raw_img, 1, mask, r, 0)
            save_img_path = os.path.join(
                mask_base_path, 'img_add_masks', video_name)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_img_path, img_name), raw_img_mask)
        count += 1
    print(f'count: {count} videos')
