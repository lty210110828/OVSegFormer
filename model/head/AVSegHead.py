'''
[原 AVSegFormer 音频-视觉分割 Head 代码，保留但不注释]
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import build_generator
from ..utils.transformer import TransformerDecoder


class AVSegHead(nn.Module):
    """
    OVSegFormer 分割头 — 核心推理模块

    将视觉特征、组学上下文、细胞核质心坐标融合，预测每个细胞的:
        1. Objectness 分数 (是否存在有效细胞)
        2. 实例级分割掩膜

    关键设计:
        - 使用 NucleiGuidedQueryGenerator 从质心坐标生成 Query
        - 使用局部视觉注意力掩膜 (_build_local_visual_attn_mask) 限制每个 Query
          只关注其质心附近的空间 token，避免远距离噪声干扰
        - 多尺度视觉特征通过 input_proj + mask_feature_fuse 融合为统一特征图
    """
    def __init__(self,
                 in_channels,
                 num_classes,
                 embed_dim=256,
                 query_num=10,
                 num_layers=3,
                 num_heads=8,
                 multi_scale_indices=None,
                 local_visual_radius=0.18,
                 local_visual_soft_bias_scale=2.0,
                 local_visual_pool_radius=0.18,
                 local_visual_pool_type='mean',
                 query_generator=None,
                 transformer=None,
                 **kwargs):
        super(AVSegHead, self).__init__()

        if transformer is not None and 'd_model' in transformer:
            if transformer['d_model'] != embed_dim:
                print(f"Warning: Config mismatch! Head embed_dim({embed_dim}) != Transformer d_model({transformer['d_model']})")

        # 确定参与融合的多尺度特征层级索引
        if isinstance(in_channels, list):
            if multi_scale_indices is None:
                multi_scale_indices = list(range(max(0, len(in_channels) - 3), len(in_channels)))
            self.multi_scale_indices = sorted([idx for idx in multi_scale_indices if 0 <= idx < len(in_channels)])
            if len(self.multi_scale_indices) == 0:
                self.multi_scale_indices = [len(in_channels) - 1]
            selected_in_channels = [in_channels[idx] for idx in self.multi_scale_indices]
        else:
            self.multi_scale_indices = [0]
            selected_in_channels = [in_channels]

        self.in_channels = selected_in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.query_num = query_num
        self.local_visual_radius = float(local_visual_radius)
        self.local_visual_soft_bias_scale = float(local_visual_soft_bias_scale)
        self.local_visual_pool_radius = float(local_visual_pool_radius)
        self.local_visual_pool_type = str(local_visual_pool_type)

        # Query 生成器: 从质心坐标映射到 query embedding
        self.query_generator = build_generator(**query_generator)

        # Transformer 解码器: query 通过交叉注意力与视觉 memory tokens 交互
        self.transformer_decoder = TransformerDecoder(**transformer)

        # 分类头: 预测每个 query 是否对应真实细胞 (objectness logit)
        self.class_embed = nn.Linear(embed_dim, 1)

        # 掩膜嵌入头: 3 层 MLP，预测每个 query 的掩膜权重向量
        self.mask_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 每个尺度一个 1×1 卷积，将 backbone 通道数映射到 embed_dim
        self.input_proj = nn.ModuleList([
            nn.Conv2d(cur_in_channels, embed_dim, kernel_size=1)
            for cur_in_channels in selected_in_channels
        ])

        # 多尺度特征融合: concat 后 1×1 卷积降维 + 3×3 卷积精炼
        self.mask_feature_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(selected_in_channels), embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )
        self.query_context_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.query_context_norm = nn.LayerNorm(embed_dim)

    def _build_memory_token_coords(self, projected_feats):
        """
        为每个视觉 memory token 计算其归一化 (x, y) 坐标

        对每层特征图生成网格坐标，用于后续构建局部注意力掩膜
        坐标范围 [0, 1]，(0.5/h, 0.5/w) 表示左上角 token

        Args:
            projected_feats: list of [B, embed_dim, H_i, W_i] 各尺度投影后特征

        Returns:
            token_coords: [total_tokens, 2] 所有 token 的归一化坐标
        """
        coords = []
        for feat in projected_feats:
            h, w = feat.shape[-2:]
            ys = (torch.arange(h, device=feat.device, dtype=feat.dtype) + 0.5) / max(h, 1)
            xs = (torch.arange(w, device=feat.device, dtype=feat.dtype) + 0.5) / max(w, 1)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            coords.append(torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1))
        return torch.cat(coords, dim=0)

    def _build_local_visual_attn_mask(self, centroids, token_coords, query_valid_mask=None):
        radius = max(self.local_visual_radius, 1e-4)
        dx = (centroids[..., 0].unsqueeze(-1) - token_coords[:, 0].view(1, 1, -1)).abs()
        dy = (centroids[..., 1].unsqueeze(-1) - token_coords[:, 1].view(1, 1, -1)).abs()
        visible = (dx <= radius) & (dy <= radius)
        if query_valid_mask is not None:
            visible = visible & query_valid_mask.unsqueeze(-1)
        no_visible = ~visible.any(dim=-1)
        if no_visible.any():
            dist2 = dx.pow(2) + dy.pow(2)
            nearest_idx = dist2.argmin(dim=-1)
            visible = visible.clone()
            visible[no_visible] = False
            visible[no_visible, nearest_idx[no_visible]] = True

        if query_valid_mask is not None:
            invalid = ~query_valid_mask
            if invalid.any():
                visible = visible.clone()
                visible[invalid] = True
        attn_mask = torch.zeros_like(visible, dtype=token_coords.dtype)
        attn_mask = attn_mask.masked_fill(~visible, -1e4)
        num_heads = self.transformer_decoder.layers[0].visual_cross_attn.num_heads
        attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
        return attn_mask.view(-1, attn_mask.shape[-2], attn_mask.shape[-1])

    def _build_local_visual_soft_bias(self, centroids, token_coords, query_valid_mask=None):
        radius = max(self.local_visual_radius, 1e-4)
        dx = centroids[..., 0].unsqueeze(-1) - token_coords[:, 0].view(1, 1, -1)
        dy = centroids[..., 1].unsqueeze(-1) - token_coords[:, 1].view(1, 1, -1)
        dist2 = (dx / radius).pow(2) + (dy / radius).pow(2)
        bias = -self.local_visual_soft_bias_scale * dist2
        if query_valid_mask is not None:
            bias = bias.masked_fill(~query_valid_mask.unsqueeze(-1), 0.0)
        num_heads = self.transformer_decoder.layers[0].visual_cross_attn.num_heads
        bias = bias.unsqueeze(1).repeat(1, num_heads, 1, 1)
        return bias.view(-1, bias.shape[-2], bias.shape[-1])

    def _pool_local_visual(self, src, centroids, query_valid_mask=None):
        batch_size, channels, height, width = src.shape
        radius_x = max(1, int(round(self.local_visual_pool_radius * width)))
        radius_y = max(1, int(round(self.local_visual_pool_radius * height)))
        x_coord = centroids[..., 0].mul(width).sub(0.5).clamp(0, width - 1)
        y_coord = centroids[..., 1].mul(height).sub(0.5).clamp(0, height - 1)
        x_center = x_coord.round().to(dtype=torch.long)
        y_center = y_coord.round().to(dtype=torch.long)

        y_grid = torch.arange(height, device=src.device, dtype=src.dtype).view(1, 1, height, 1)
        x_grid = torch.arange(width, device=src.device, dtype=src.dtype).view(1, 1, 1, width)
        window_mask = (x_grid - x_coord.unsqueeze(-1).unsqueeze(-1)).abs() <= float(radius_x)
        window_mask = window_mask & ((y_grid - y_coord.unsqueeze(-1).unsqueeze(-1)).abs() <= float(radius_y))
        if query_valid_mask is not None:
            window_mask = window_mask & query_valid_mask.unsqueeze(-1).unsqueeze(-1)

        center_feat = src.permute(0, 2, 3, 1)[
            torch.arange(batch_size, device=src.device).view(-1, 1),
            y_center,
            x_center,
        ]

        mask_float = window_mask.unsqueeze(2).to(dtype=src.dtype)
        if self.local_visual_pool_type == 'max':
            pooled = src.unsqueeze(1).masked_fill(~window_mask.unsqueeze(2), torch.finfo(src.dtype).min).amax(dim=(-2, -1))
            empty_mask = ~window_mask.flatten(2).any(dim=-1)
            if empty_mask.any():
                pooled = torch.where(empty_mask.unsqueeze(-1), center_feat, pooled)
        else:
            denom = mask_float.sum(dim=(-2, -1)).clamp_min(1.0)
            pooled = (src.unsqueeze(1) * mask_float).sum(dim=(-2, -1)) / denom
            empty_mask = ~window_mask.flatten(2).any(dim=-1)
            if empty_mask.any():
                pooled = torch.where(empty_mask.unsqueeze(-1), center_feat, pooled)

        if query_valid_mask is not None:
            pooled = pooled.masked_fill(~query_valid_mask.unsqueeze(-1), 0.0)
        return pooled

    def forward(self, x, omics_context, centroids=None, query_valid_mask=None):
        """
        前向传播

        Args:
            x: 视觉特征 (list of [B,C,H,W] 或 单个 [B,C,H,W])
            omics_context: [B, N_queries, embed_dim] 组学上下文特征
            centroids: [B, N_queries, 2] 细胞质心归一化坐标
            query_valid_mask: [B, N_queries] 有效 query 布尔掩膜

        Returns:
            outputs_class: [B, N_queries, 1] objectness logits
            outputs_mask: [B, N_queries, H, W] 实例分割掩膜 logits
        """
        # Step 1: 多尺度视觉特征投影 + 展平为 token 序列
        if isinstance(x, list):
            visual_feats = [x[idx] for idx in self.multi_scale_indices]
        else:
            visual_feats = [x]

        projected_feats = []
        memory_tokens = []
        for feat, proj in zip(visual_feats, self.input_proj):
            cur_feat = proj(feat)
            projected_feats.append(cur_feat)
            memory_tokens.append(cur_feat.flatten(2).transpose(1, 2))  # [B, C, H*W] → [B, H*W, C]

        # 将多尺度特征 resize 到统一尺寸后 concat，再融合为单一特征图 src
        fuse_h, fuse_w = projected_feats[0].shape[-2:]
        resized_feats = []
        for feat in projected_feats:
            if feat.shape[-2:] != (fuse_h, fuse_w):
                feat = F.interpolate(feat, size=(fuse_h, fuse_w), mode='bilinear', align_corners=False)
            resized_feats.append(feat)
        src = self.mask_feature_fuse(torch.cat(resized_feats, dim=1))  # [B, embed_dim, H, W]
        src_flatten = torch.cat(memory_tokens, dim=1)  # [B, total_tokens, embed_dim]

        token_coords = self._build_memory_token_coords(projected_feats)

        # Step 2: 从质心坐标生成 Query embeddings
        if centroids is not None:
            query_embed = self.query_generator(centroids)  # [B, N_queries, embed_dim]
        else:
            query_embed = self.query_generator(omics_context)

        visual_attn_mask = None
        visual_attn_bias = None
        if centroids is not None:
            visual_attn_mask = self._build_local_visual_attn_mask(centroids, token_coords, query_valid_mask=query_valid_mask)
            visual_attn_bias = self._build_local_visual_soft_bias(centroids, token_coords, query_valid_mask=query_valid_mask)
            query_local_visual = self._pool_local_visual(src, centroids, query_valid_mask=query_valid_mask)
            query_context = self.query_context_proj(torch.cat([query_embed, omics_context, query_local_visual], dim=-1))
            query_embed = self.query_context_norm(query_embed + query_context)

        out = self.transformer_decoder(
            query_embed,
            src_flatten,
            omics_context,
            visual_attn_mask=visual_attn_mask,
            visual_attn_bias=visual_attn_bias,
        )

        # Step 4: 预测分类分数和掩膜
        outputs_class = self.class_embed(out)  # [B, N_queries, 1]

        mask_embed = self.mask_embed(out)  # [B, N_queries, embed_dim]

        # 掩膜 = mask_embed @ visual_features^T → [B, N_queries, H, W]
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, src)

        return outputs_class, outputs_mask
