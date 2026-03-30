from utils.pyutils import AverageMeter
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class IouSemanticAwareLoss(nn.Module):
    """
    真正的 IouSemanticAwareLoss 实现
    结合了 BCE (Binary Cross Entropy) 和 IoU / Dice Loss，
    能更好地处理类别不平衡问题，并关注区域的重合度（Semantic Aware）。
    """
    def __init__(self, bce_weight=1.0, iou_weight=1.0, dice_weight=0.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.dice_weight = dice_weight
        
    def forward(self, pred_logits, target_mask):
        """
        pred_logits: [B, 1, H, W] 模型输出的对数几率 (未经过 sigmoid)
        target_mask: [B, 1, H, W] Ground Truth 掩膜 (0 或 1)
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. BCE Loss (关注像素级别的分类)
        if self.bce_weight > 0:
            bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_mask)
            total_loss += self.bce_weight * bce_loss
            loss_dict['bce_loss'] = bce_loss.item()
            
        # 准备计算区域损失
        pred_probs = torch.sigmoid(pred_logits)
        pred_flat = pred_probs.flatten(1)
        target_flat = target_mask.flatten(1)
        
        intersection = (pred_flat * target_flat).sum(-1)
        
        # 2. IoU Loss (关注交并比，感知语义整体性)
        if self.iou_weight > 0:
            union = pred_flat.sum(-1) + target_flat.sum(-1) - intersection
            iou = (intersection + 1e-5) / (union + 1e-5)
            iou_loss = (1.0 - iou).mean()
            total_loss += self.iou_weight * iou_loss
            loss_dict['iou_loss'] = iou_loss.item()
            
        # 3. Dice Loss (可选，类似于 F1 分数的平滑版本)
        if self.dice_weight > 0:
            # 使用平方可以使梯度更加平滑
            union_dice = (pred_flat * pred_flat).sum(-1) + (target_flat * target_flat).sum(-1)
            dice = (2.0 * intersection + 1e-5) / (union_dice + 1e-5)
            dice_loss = (1.0 - dice).mean()
            total_loss += self.dice_weight * dice_loss
            loss_dict['dice_loss'] = dice_loss.item()
            
        return total_loss, loss_dict


class AlignedInstanceSegLoss(nn.Module):
    def __init__(
        self,
        cls_weight=0.0,
        mask_bce_weight=1.0,
        mask_dice_weight=1.0,
        overlap_weight=0.0,
        mask_loss_size=128,
        overlap_loss_size=64,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.mask_bce_weight = mask_bce_weight
        self.mask_dice_weight = mask_dice_weight
        self.overlap_weight = overlap_weight
        self.mask_loss_size = mask_loss_size
        self.overlap_loss_size = overlap_loss_size

    def _resize_masks(self, pred_masks, target_masks):
        if pred_masks.shape[-2:] == target_masks.shape[-2:]:
            return pred_masks
        batch_queries = pred_masks.shape[0] * pred_masks.shape[1]
        pred_masks = pred_masks.reshape(batch_queries, 1, pred_masks.shape[-2], pred_masks.shape[-1])
        pred_masks = F.interpolate(pred_masks, size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        return pred_masks.reshape(target_masks.shape[0], target_masks.shape[1], target_masks.shape[-2], target_masks.shape[-1])

    def _dice_loss(self, pred_logits, target_masks):
        pred_probs = torch.sigmoid(pred_logits)
        pred_flat = pred_probs.flatten(1)
        target_flat = target_masks.flatten(1)
        intersection = (pred_flat * target_flat).sum(-1)
        union = pred_flat.sum(-1) + target_flat.sum(-1)
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        return 1.0 - dice

    def _downsample_pair_masks(self, pred_masks, target_masks):
        if self.mask_loss_size is None:
            return pred_masks, target_masks
        target_size = (int(self.mask_loss_size), int(self.mask_loss_size))
        if pred_masks.shape[-2:] != target_size:
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(1),
                size=target_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        if target_masks.shape[-2:] != target_size:
            target_masks = F.interpolate(
                target_masks.unsqueeze(1),
                size=target_size,
                mode='nearest',
            ).squeeze(1)
        return pred_masks, target_masks

    def _overlap_loss(self, pred_masks, valid_mask):
        pred_probs = torch.sigmoid(pred_masks)
        if self.overlap_loss_size is not None and pred_probs.shape[-2:] != (int(self.overlap_loss_size), int(self.overlap_loss_size)):
            pred_probs = F.interpolate(
                pred_probs,
                size=(int(self.overlap_loss_size), int(self.overlap_loss_size)),
                mode='bilinear',
                align_corners=False,
            )
        overlap_terms = []
        for batch_idx in range(pred_probs.shape[0]):
            keep = valid_mask[batch_idx]
            if keep.sum() <= 1:
                continue
            cur_masks = pred_probs[batch_idx][keep].flatten(1)
            areas = cur_masks.sum(-1)
            inter = cur_masks @ cur_masks.t()
            denom = torch.minimum(areas[:, None], areas[None, :]).clamp_min(1e-6)
            overlap = inter / denom
            upper = torch.triu(overlap, diagonal=1)
            pair_mask = torch.triu(torch.ones_like(overlap, dtype=torch.bool), diagonal=1)
            if pair_mask.any():
                overlap_terms.append(upper[pair_mask].mean())
        if len(overlap_terms) == 0:
            return pred_masks.new_tensor(0.0)
        return torch.stack(overlap_terms).mean()

    def forward(self, pred_cls, pred_masks, target_masks, target_valid_mask):
        loss_dict = {}
        total_loss = pred_masks.new_tensor(0.0)
        cls_loss = pred_masks.new_tensor(0.0)
        if self.cls_weight > 0:
            query_targets = target_valid_mask.float()
            cls_loss = F.binary_cross_entropy_with_logits(pred_cls.squeeze(-1), query_targets)
            total_loss = total_loss + self.cls_weight * cls_loss
        loss_dict['inst_cls_loss'] = cls_loss.item()

        pred_masks = self._resize_masks(pred_masks, target_masks)
        positive_mask = target_valid_mask
        if positive_mask.any():
            pred_pos = pred_masks[positive_mask]
            target_pos = target_masks[positive_mask]
            pred_pos, target_pos = self._downsample_pair_masks(pred_pos, target_pos)
            bce_loss = F.binary_cross_entropy_with_logits(pred_pos, target_pos)
            dice_loss = self._dice_loss(pred_pos, target_pos).mean()
        else:
            bce_loss = pred_masks.new_tensor(0.0)
            dice_loss = pred_masks.new_tensor(0.0)
        total_loss = total_loss + self.mask_bce_weight * bce_loss + self.mask_dice_weight * dice_loss
        loss_dict['inst_mask_bce_loss'] = bce_loss.item()
        loss_dict['inst_mask_dice_loss'] = dice_loss.item()

        overlap_loss = pred_masks.new_tensor(0.0)
        if self.overlap_weight > 0:
            overlap_loss = self._overlap_loss(pred_masks, target_valid_mask)
            total_loss = total_loss + self.overlap_weight * overlap_loss
        loss_dict['inst_overlap_loss'] = overlap_loss.item()
        loss_dict['inst_pos_queries'] = float(target_valid_mask.sum().item())

        return total_loss, loss_dict, None


class HungarianMatcher(nn.Module):
    def __init__(self, cls_cost=1.0, mask_cost=1.0, dice_cost=1.0):
        super().__init__()
        self.cls_cost = cls_cost
        self.mask_cost = mask_cost
        self.dice_cost = dice_cost

    def _resize_masks(self, pred_masks, target_masks):
        if pred_masks.shape[-2:] == target_masks.shape[-2:]:
            return pred_masks
        batch_size, query_num = pred_masks.shape[:2]
        pred_masks = pred_masks.reshape(batch_size * query_num, 1, pred_masks.shape[-2], pred_masks.shape[-1])
        pred_masks = F.interpolate(pred_masks, size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        return pred_masks.reshape(batch_size, query_num, target_masks.shape[-2], target_masks.shape[-1])

    def _pairwise_mask_cost(self, pred_logits, target_masks):
        pred_flat = pred_logits.flatten(1)
        target_flat = target_masks.flatten(1)
        pos_cost = F.softplus(-pred_flat) @ target_flat.t()
        neg_cost = F.softplus(pred_flat) @ (1.0 - target_flat).t()
        return (pos_cost + neg_cost) / pred_flat.shape[1]

    def _pairwise_dice_cost(self, pred_logits, target_masks, eps=1e-6):
        pred_probs = torch.sigmoid(pred_logits).flatten(1)
        target_flat = target_masks.flatten(1)
        inter = pred_probs @ target_flat.t()
        pred_sum = pred_probs.sum(-1, keepdim=True)
        target_sum = target_flat.sum(-1).unsqueeze(0)
        dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
        return 1.0 - dice

    @torch.no_grad()
    def forward(self, pred_cls, pred_masks, target_masks, target_valid_mask):
        if pred_masks.shape[-2:] != target_masks.shape[-2:]:
            batch_queries = pred_masks.shape[0] * pred_masks.shape[1]
            pred_masks = pred_masks.reshape(batch_queries, 1, pred_masks.shape[-2], pred_masks.shape[-1])
            pred_masks = F.interpolate(pred_masks, size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
            pred_masks = pred_masks.reshape(target_masks.shape[0], target_valid_mask.shape[1], target_masks.shape[-2], target_masks.shape[-1])

        indices = []
        cls_prob = torch.sigmoid(pred_cls.squeeze(-1))
        for batch_idx in range(pred_cls.shape[0]):
            tgt_mask = target_valid_mask[batch_idx].bool()
            tgt_count = int(tgt_mask.sum().item())
            if tgt_count == 0:
                empty = pred_cls.new_zeros((0,), dtype=torch.long)
                indices.append((empty, empty))
                continue
            cur_pred_masks = pred_masks[batch_idx]
            cur_tgt_masks = target_masks[batch_idx][tgt_mask].float()
            cls_cost = -cls_prob[batch_idx].unsqueeze(1).expand(-1, tgt_count)
            mask_cost = self._pairwise_mask_cost(cur_pred_masks, cur_tgt_masks)
            dice_cost = self._pairwise_dice_cost(cur_pred_masks, cur_tgt_masks)
            total_cost = self.cls_cost * cls_cost + self.mask_cost * mask_cost + self.dice_cost * dice_cost
            row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long, device=pred_cls.device),
                torch.as_tensor(col_ind, dtype=torch.long, device=pred_cls.device),
            ))
        return indices


class HungarianInstanceSegLoss(nn.Module):
    def __init__(
        self,
        cls_weight=1.0,
        mask_bce_weight=1.0,
        mask_dice_weight=1.0,
        overlap_weight=0.0,
        no_object_weight=0.1,
        matcher_cls_cost=1.0,
        matcher_mask_cost=1.0,
        matcher_dice_cost=1.0,
        mask_loss_size=128,
        overlap_loss_size=64,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.mask_bce_weight = mask_bce_weight
        self.mask_dice_weight = mask_dice_weight
        self.overlap_weight = overlap_weight
        self.no_object_weight = no_object_weight
        self.mask_loss_size = mask_loss_size
        self.overlap_loss_size = overlap_loss_size
        self.matcher = HungarianMatcher(
            cls_cost=matcher_cls_cost,
            mask_cost=matcher_mask_cost,
            dice_cost=matcher_dice_cost,
        )

    def _resize_masks(self, pred_masks, target_masks):
        if pred_masks.shape[-2:] == target_masks.shape[-2:]:
            return pred_masks
        batch_size, query_num = pred_masks.shape[:2]
        flat_masks = pred_masks.reshape(batch_size * query_num, 1, pred_masks.shape[-2], pred_masks.shape[-1])
        flat_masks = F.interpolate(flat_masks, size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        return flat_masks.reshape(batch_size, query_num, target_masks.shape[-2], target_masks.shape[-1])

    def _dice_loss(self, pred_logits, target_masks):
        pred_probs = torch.sigmoid(pred_logits).flatten(1)
        target_flat = target_masks.flatten(1)
        intersection = (pred_probs * target_flat).sum(-1)
        denom = pred_probs.sum(-1) + target_flat.sum(-1)
        dice = (2.0 * intersection + 1e-5) / (denom + 1e-5)
        return 1.0 - dice

    def _downsample_pair_masks(self, pred_masks, target_masks):
        if self.mask_loss_size is None:
            return pred_masks, target_masks
        target_size = (int(self.mask_loss_size), int(self.mask_loss_size))
        if pred_masks.shape[-2:] != target_size:
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(1),
                size=target_size,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        if target_masks.shape[-2:] != target_size:
            target_masks = F.interpolate(
                target_masks.unsqueeze(1),
                size=target_size,
                mode='nearest',
            ).squeeze(1)
        return pred_masks, target_masks

    def _overlap_loss(self, pred_masks, positive_mask):
        pred_probs = torch.sigmoid(pred_masks)
        if self.overlap_loss_size is not None and pred_probs.shape[-2:] != (int(self.overlap_loss_size), int(self.overlap_loss_size)):
            pred_probs = F.interpolate(
                pred_probs,
                size=(int(self.overlap_loss_size), int(self.overlap_loss_size)),
                mode='bilinear',
                align_corners=False,
            )
        overlap_terms = []
        for batch_idx in range(pred_probs.shape[0]):
            keep = positive_mask[batch_idx]
            if keep.sum() <= 1:
                continue
            cur_masks = pred_probs[batch_idx][keep].flatten(1)
            areas = cur_masks.sum(-1)
            inter = cur_masks @ cur_masks.t()
            denom = torch.minimum(areas[:, None], areas[None, :]).clamp_min(1e-6)
            overlap = inter / denom
            pair_mask = torch.triu(torch.ones_like(overlap, dtype=torch.bool), diagonal=1)
            if pair_mask.any():
                overlap_terms.append(overlap[pair_mask].mean())
        if len(overlap_terms) == 0:
            return pred_masks.new_tensor(0.0)
        return torch.stack(overlap_terms).mean()

    def forward(self, pred_cls, pred_masks, target_masks, target_valid_mask):
        pred_masks = self._resize_masks(pred_masks, target_masks)
        matched_indices = self.matcher(pred_cls, pred_masks, target_masks, target_valid_mask)

        cls_targets = pred_cls.new_zeros(pred_cls.shape[:2])
        positive_mask = torch.zeros_like(cls_targets, dtype=torch.bool)
        matched_pred_masks = []
        matched_target_masks = []

        for batch_idx, (pred_idx, tgt_idx) in enumerate(matched_indices):
            if pred_idx.numel() == 0:
                continue
            cls_targets[batch_idx, pred_idx] = 1.0
            positive_mask[batch_idx, pred_idx] = True
            valid_target_masks = target_masks[batch_idx][target_valid_mask[batch_idx].bool()].float()
            matched_pred_masks.append(pred_masks[batch_idx, pred_idx])
            matched_target_masks.append(valid_target_masks[tgt_idx])

        cls_weights = pred_cls.new_full(cls_targets.shape, self.no_object_weight)
        cls_weights[cls_targets > 0] = 1.0
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls.squeeze(-1), cls_targets, weight=cls_weights)

        if len(matched_pred_masks) > 0:
            matched_pred_masks = torch.cat(matched_pred_masks, dim=0)
            matched_target_masks = torch.cat(matched_target_masks, dim=0)
            matched_pred_masks, matched_target_masks = self._downsample_pair_masks(
                matched_pred_masks,
                matched_target_masks,
            )
            mask_bce_loss = F.binary_cross_entropy_with_logits(matched_pred_masks, matched_target_masks)
            mask_dice_loss = self._dice_loss(matched_pred_masks, matched_target_masks).mean()
        else:
            mask_bce_loss = pred_masks.new_tensor(0.0)
            mask_dice_loss = pred_masks.new_tensor(0.0)

        overlap_loss = pred_masks.new_tensor(0.0)
        if self.overlap_weight > 0:
            overlap_loss = self._overlap_loss(pred_masks, positive_mask)

        total_loss = (
            self.cls_weight * cls_loss
            + self.mask_bce_weight * mask_bce_loss
            + self.mask_dice_weight * mask_dice_loss
            + self.overlap_weight * overlap_loss
        )
        loss_dict = {
            'inst_cls_loss': cls_loss.item(),
            'inst_mask_bce_loss': mask_bce_loss.item(),
            'inst_mask_dice_loss': mask_dice_loss.item(),
            'inst_overlap_loss': overlap_loss.item(),
            'inst_pos_queries': float(positive_mask.sum().item()),
        }
        return total_loss, loss_dict, matched_indices


class LossUtil:
    def __init__(self, weight_dict, **kwargs) -> None:
        self.loss_weight_dict = weight_dict
        self.avg_loss = dict()
        self.avg_loss['total_loss'] = AverageMeter('total_loss')
        # for k in weight_dict.keys():
        #     self.avg_loss[k] = AverageMeter(k)

    def add_loss(self, loss, loss_dict):
        self.avg_loss['total_loss'].add({'total_loss': loss.item()})
        for k, v in loss_dict.items():
            meter = self.avg_loss.get(k, None)
            if meter is None:
                meter = AverageMeter(k)
                self.avg_loss[k] = meter

            self.avg_loss[k].add({k: v})

    def pretty_out(self):
        f = 'Total_Loss:%.4f, ' % (
            self.avg_loss['total_loss'].pop('total_loss'))
        for k in self.avg_loss.keys():
            if k == 'total_loss':
                continue
            t = '%s:%.4f, ' % (k, self.avg_loss[k].pop(k))
            f += t
        return f
