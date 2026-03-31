from utils.pyutils import AverageMeter
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# ============================================================
# [原 AVSegFormer 语义级损失 — 保留但不注释]
# 用于像素级前景/背景分割的辅助损失 (BCE + IoU + Dice)
# OVSegFormer 实例分割中不使用此类
# ============================================================

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


# ============================================================
# [OVSegFormer 实例级损失函数]
# ============================================================

class AlignedInstanceSegLoss(nn.Module):
    """
    对齐式实例分割损失 — 直接按 query 位置与 GT 一一对应，无需 Hungarian 匹配

    核心思路:
        每个 query 由细胞质心生成，天然与对应的 GT 实例对齐，
        因此直接对正样本 query 计算 mask BCE + Dice 损失。
        适用于训练初期或 query-GT 一一对应的场景。

    损失组成:
        1. cls_loss:    目标性分类损失 (BCE)，区分有效/无效 query
        2. mask_bce:    掩膜二值交叉熵 (仅对正样本 query)
        3. mask_dice:   掩膜 Dice 损失 (仅对正样本 query)
        4. overlap:     实例间重叠惩罚，抑制预测掩膜的重叠区域

    Args:
        cls_weight:        分类损失权重 (默认 0.0 表示不使用)
        mask_bce_weight:   掩膜 BCE 权重
        mask_dice_weight:  掩膜 Dice 权重
        overlap_weight:    重叠惩罚权重
        mask_loss_size:    计算掩膜损失时的下采样尺寸 (节省显存)
        overlap_loss_size: 计算重叠损失时的下采样尺寸
    """
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
        """将预测掩膜双线性插值到 GT 掩膜尺寸 — [B, N, H, W]"""
        if pred_masks.shape[-2:] == target_masks.shape[-2:]:
            return pred_masks
        batch_queries = pred_masks.shape[0] * pred_masks.shape[1]
        pred_masks = pred_masks.reshape(batch_queries, 1, pred_masks.shape[-2], pred_masks.shape[-1])
        pred_masks = F.interpolate(pred_masks, size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        return pred_masks.reshape(target_masks.shape[0], target_masks.shape[1], target_masks.shape[-2], target_masks.shape[-1])

    def _dice_loss(self, pred_logits, target_masks):
        """计算 Dice 损失: 1 - (2*交集)/(pred+target)"""
        pred_probs = torch.sigmoid(pred_logits)
        pred_flat = pred_probs.flatten(1)
        target_flat = target_masks.flatten(1)
        intersection = (pred_flat * target_flat).sum(-1)
        union = pred_flat.sum(-1) + target_flat.sum(-1)
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        return 1.0 - dice

    def _downsample_pair_masks(self, pred_masks, target_masks):
        """
        将掩膜下采样到 mask_loss_size×mask_loss_size — 大幅减少显存占用
        预测掩膜用双线性插值，GT 掩膜用最近邻插值 (保持二值性)
        """
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
        """
        计算实例间重叠惩罚 — 鼓励不同 query 的预测掩膜尽量不重叠

        对每个 batch，取所有有效 query 的预测掩膜，
        计算两两之间的重叠率 (交集/较小面积)，
        取上三角均值作为 batch 的重叠损失。
        """
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
        """
        前向计算对齐式实例分割损失

        Args:
            pred_cls:          [B, N, 1] query 目标性 logits
            pred_masks:        [B, N, H', W'] query 掩膜 logits
            target_masks:      [B, N_gt_max, H, W] GT 实例掩膜 (已 padding)
            target_valid_mask: [B, N_gt_max] 有效实例标记

        Returns:
            total_loss: 加权总损失
            loss_dict:  各项损失的标量值 (用于日志记录)
            None:       对齐损失不需要返回匹配索引
        """
        loss_dict = {}
        total_loss = pred_masks.new_tensor(0.0)

        # 1. 分类损失: 区分有效 query (有对应 GT 细胞) vs 无效 query (padding)
        cls_loss = pred_masks.new_tensor(0.0)
        if self.cls_weight > 0:
            query_targets = target_valid_mask.float()
            cls_loss = F.binary_cross_entropy_with_logits(pred_cls.squeeze(-1), query_targets)
            total_loss = total_loss + self.cls_weight * cls_loss
        loss_dict['inst_cls_loss'] = cls_loss.item()

        # 2. 掩膜损失: 仅对正样本 query 计算 BCE + Dice
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

        # 3. 重叠惩罚: 抑制不同 query 的预测掩膜互相重叠
        overlap_loss = pred_masks.new_tensor(0.0)
        if self.overlap_weight > 0:
            overlap_loss = self._overlap_loss(pred_masks, target_valid_mask)
            total_loss = total_loss + self.overlap_weight * overlap_loss
        loss_dict['inst_overlap_loss'] = overlap_loss.item()
        loss_dict['inst_pos_queries'] = float(target_valid_mask.sum().item())

        return total_loss, loss_dict, None


class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器 — 为每个 GT 实例找到最优的 query 分配

    代价矩阵由三部分加权求和:
        1. cls_cost:  分类概率代价 (负 sigmoid 值，倾向匹配高分 query)
        2. mask_cost: 掩膜像素级 BCE 代价
        3. dice_cost: 掩膜 Dice 代价

    使用 scipy.optimize.linear_sum_assignment 求解最优二部图匹配。
    """
    def __init__(self, cls_cost=1.0, mask_cost=1.0, dice_cost=1.0):
        super().__init__()
        self.cls_cost = cls_cost
        self.mask_cost = mask_cost
        self.dice_cost = dice_cost

    def _resize_masks(self, pred_masks, target_masks):
        """将预测掩膜插值到 GT 尺寸 — [B, N, H, W]"""
        if pred_masks.shape[-2:] == target_masks.shape[-2:]:
            return pred_masks
        batch_size, query_num = pred_masks.shape[:2]
        pred_masks = pred_masks.reshape(batch_size * query_num, 1, pred_masks.shape[-2], pred_masks.shape[-1])
        pred_masks = F.interpolate(pred_masks, size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        return pred_masks.reshape(batch_size, query_num, target_masks.shape[-2], target_masks.shape[-1])

    def _pairwise_mask_cost(self, pred_logits, target_masks):
        """
        计算预测-目标掩膜的逐对 BCE 代价矩阵

        Returns:
            cost: [N_pred, N_gt] 每对 (pred, gt) 的平均像素 BCE 代价
        """
        pred_flat = pred_logits.flatten(1)
        target_flat = target_masks.flatten(1)
        # 正像素代价: softplus(-logit) * target
        pos_cost = F.softplus(-pred_flat) @ target_flat.t()
        # 负像素代价: softplus(logit) * (1-target)
        neg_cost = F.softplus(pred_flat) @ (1.0 - target_flat).t()
        return (pos_cost + neg_cost) / pred_flat.shape[1]

    def _pairwise_dice_cost(self, pred_logits, target_masks, eps=1e-6):
        """
        计算预测-目标掩膜的逐对 Dice 代价矩阵

        Returns:
            cost: [N_pred, N_gt] 每对的 1 - Dice
        """
        pred_probs = torch.sigmoid(pred_logits).flatten(1)
        target_flat = target_masks.flatten(1)
        inter = pred_probs @ target_flat.t()
        pred_sum = pred_probs.sum(-1, keepdim=True)
        target_sum = target_flat.sum(-1).unsqueeze(0)
        dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
        return 1.0 - dice

    @torch.no_grad()
    def forward(self, pred_cls, pred_masks, target_masks, target_valid_mask):
        """
        执行匈牙利匹配

        Args:
            pred_cls:          [B, N, 1] query 分类 logits
            pred_masks:        [B, N, H', W'] 预测掩膜 logits
            target_masks:      [B, N_gt_max, H, W] GT 实例掩膜
            target_valid_mask: [B, N_gt_max] 有效 GT 标记

        Returns:
            indices: list of (pred_indices, gt_indices) 元组，每个 batch 一组
        """
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
            # 构建代价矩阵: cls + mask_bce + dice
            cls_cost = -cls_prob[batch_idx].unsqueeze(1).expand(-1, tgt_count)
            mask_cost = self._pairwise_mask_cost(cur_pred_masks, cur_tgt_masks)
            dice_cost = self._pairwise_dice_cost(cur_pred_masks, cur_tgt_masks)
            total_cost = self.cls_cost * cls_cost + self.mask_cost * mask_cost + self.dice_cost * dice_cost
            # Hungarian 最优匹配
            row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long, device=pred_cls.device),
                torch.as_tensor(col_ind, dtype=torch.long, device=pred_cls.device),
            ))
        return indices


class HungarianInstanceSegLoss(nn.Module):
    """
    匈牙利匹配式实例分割损失 — 通过 Hungarian 最优匹配关联 query 和 GT

    与 AlignedInstanceSegLoss 的区别:
        - Aligned: query 与 GT 按位置直接对齐 (适用于一一对应场景)
        - Hungarian: 通过代价矩阵做最优匹配 (适用于一对多/多对一场景，更鲁棒)

    损失组成:
        1. cls_loss:    加权分类损失 (正样本权重=1.0，负样本权重=no_object_weight)
        2. mask_bce:    匹配对的掩膜 BCE 损失
        3. mask_dice:   匹配对的掩膜 Dice 损失
        4. overlap:     实例间重叠惩罚

    Args:
        cls_weight:           分类损失权重
        mask_bce_weight:      掩膜 BCE 权重
        mask_dice_weight:     掩膜 Dice 权重
        overlap_weight:       重叠惩罚权重
        no_object_weight:     负样本 (无对应 GT 的 query) 的 BCE 权重
        matcher_cls_cost:     匹配器中分类代价系数
        matcher_mask_cost:    匹配器中掩膜代价系数
        matcher_dice_cost:    匹配器中 Dice 代价系数
        mask_loss_size:       掩膜损失下采样尺寸
        overlap_loss_size:    重叠损失下采样尺寸
    """
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
        """将预测掩膜插值到 GT 尺寸 — [B, N, H, W]"""
        if pred_masks.shape[-2:] == target_masks.shape[-2:]:
            return pred_masks
        batch_size, query_num = pred_masks.shape[:2]
        flat_masks = pred_masks.reshape(batch_size * query_num, 1, pred_masks.shape[-2], pred_masks.shape[-1])
        flat_masks = F.interpolate(flat_masks, size=target_masks.shape[-2:], mode='bilinear', align_corners=False)
        return flat_masks.reshape(batch_size, query_num, target_masks.shape[-2], target_masks.shape[-1])

    def _dice_loss(self, pred_logits, target_masks):
        """计算 Dice 损失: 1 - (2*交集)/(pred+target)"""
        pred_probs = torch.sigmoid(pred_logits).flatten(1)
        target_flat = target_masks.flatten(1)
        intersection = (pred_probs * target_flat).sum(-1)
        denom = pred_probs.sum(-1) + target_flat.sum(-1)
        dice = (2.0 * intersection + 1e-5) / (denom + 1e-5)
        return 1.0 - dice

    def _downsample_pair_masks(self, pred_masks, target_masks):
        """
        将匹配对的掩膜下采样到 mask_loss_size — 减少显存
        预测用双线性插值，GT 用最近邻 (保持二值)
        """
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
        """
        计算被匹配为正样本的 query 之间的掩膜重叠惩罚

        与 AlignedInstanceSegLoss 中的重叠损失类似，
        但此处只对 Hungarian 匹配判定为正样本的 query 计算。
        """
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
        """
        前向计算匈牙利匹配式实例分割损失

        Args:
            pred_cls:          [B, N, 1] query 分类 logits
            pred_masks:        [B, N, H', W'] query 掩膜 logits
            target_masks:      [B, N_gt_max, H, W] GT 实例掩膜
            target_valid_mask: [B, N_gt_max] 有效 GT 标记

        Returns:
            total_loss:      加权总损失
            loss_dict:       各项损失的标量值
            matched_indices: 匹配结果 list of (pred_idx, gt_idx)
        """
        # 将预测掩膜 resize 到 GT 尺寸
        pred_masks = self._resize_masks(pred_masks, target_masks)

        # Step 1: 匈牙利匹配 — 找到每个 GT 实例的最优 query 分配
        matched_indices = self.matcher(pred_cls, pred_masks, target_masks, target_valid_mask)

        # Step 2: 根据匹配结果构建分类目标
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

        # Step 3: 加权分类损失 — 负样本权重较低，防止淹没正样本梯度
        cls_weights = pred_cls.new_full(cls_targets.shape, self.no_object_weight)
        cls_weights[cls_targets > 0] = 1.0
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls.squeeze(-1), cls_targets, weight=cls_weights)

        # Step 4: 匹配对的掩膜损失 (BCE + Dice)，下采样后计算
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

        # Step 5: 重叠惩罚
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


# ============================================================
# [原 AVSegFormer 损失记录工具 — 保留但不注释]
# ============================================================

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
