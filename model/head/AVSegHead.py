'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import build_transformer, build_positional_encoding, build_fusion_block, build_generator
from ops.modules import MSDeformAttn
from torch.nn.init import normal_
from torch.nn.functional import interpolate


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()

        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Conv2d(n, k, kernel_size=1, stride=1, padding=0)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SimpleFPN(nn.Module):
    def __init__(self, channel=256, layers=3):
        super().__init__()

        assert layers == 3  # only designed for 3 layers
        self.up1 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        self.up2 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        self.up3 = nn.Sequential(
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.up1(x[-1])
        x1 = x1 + x[-2]

        x2 = self.up2(x1)
        x2 = x2 + x[-3]

        y = self.up3(x2)
        return y


class AVSegHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 query_num,
                 transformer,
                 query_generator,
                 embed_dim=256,
                 valid_indices=[1, 2, 3],
                 scale_factor=4,
                 positional_encoding=None,
                 use_learnable_queries=True,
                 fusion_block=None) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num
        self.valid_indices = valid_indices
        self.num_feats = len(valid_indices)
        self.scale_factor = scale_factor
        self.use_learnable_queries = use_learnable_queries
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feats, embed_dim))
        self.learnable_query = nn.Embedding(query_num, embed_dim)

        self.query_generator = build_generator(**query_generator)

        self.transformer = build_transformer(**transformer)
        if positional_encoding is not None:
            self.positional_encoding = build_positional_encoding(
                **positional_encoding)
        else:
            self.positional_encoding = None

        in_proj = []
        for c in in_channels:
            in_proj.append(
                nn.Sequential(
                    nn.Conv2d(c, embed_dim, kernel_size=1),
                    nn.GroupNorm(32, embed_dim)
                )
            )
        self.in_proj = nn.ModuleList(in_proj)
        self.mlp = MLP(query_num, 2048, embed_dim, 3)

        if fusion_block is not None:
            self.fusion_block = build_fusion_block(**fusion_block)

        self.lateral_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, embed_dim)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.ReLU(True)
        )

        self.fpn = SimpleFPN()
        self.attn_fc = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=scale_factor, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.fc = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=scale_factor, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def reform_output_squences(self, memory, spatial_shapes, level_start_index, dim=1):
        split_size_or_sections = [None] * self.num_feats
        for i in range(self.num_feats):
            if i < self.num_feats - 1:
                split_size_or_sections[i] = level_start_index[i +
                                                              1] - level_start_index[i]
            else:
                split_size_or_sections[i] = memory.shape[dim] - \
                    level_start_index[i]
        y = torch.split(memory, split_size_or_sections, dim=dim)
        return y

    def forward(self, feats, audio_feat):
        feat14 = self.in_proj[0](feats[0])
        srcs = [self.in_proj[i](feats[i]) for i in self.valid_indices]
        masks = [torch.zeros((x.size(0), x.size(2), x.size(
            3)), device=x.device, dtype=torch.bool) for x in srcs]
        pos_embeds = []
        for m in masks:
            pos_embeds.append(self.positional_encoding(m))
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # prepare queries
        bs = audio_feat.shape[0]
        query = self.query_generator(audio_feat)
        if self.use_learnable_queries:
            query = query + \
                self.learnable_query.weight[None, :, :].repeat(bs, 1, 1)

        memory, outputs = self.transformer(query, src_flatten, spatial_shapes,
                                           level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # generate mask feature
        mask_feats = []
        for i, z in enumerate(self.reform_output_squences(memory, spatial_shapes, level_start_index, 1)):
            mask_feats.append(z.transpose(1, 2).view(
                bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        cur_fpn = self.lateral_conv(feat14)
        mask_feature = mask_feats[0]
        mask_feature = cur_fpn + \
            F.interpolate(
                mask_feature, size=cur_fpn.shape[-2:], mode='bilinear', align_corners=False)
        mask_feature = self.out_conv(mask_feature)
        if hasattr(self, 'fusion_block'):
            mask_feature = self.fusion_block(mask_feature, audio_feat)

        # predict output mask
        pred_feature = torch.einsum(
            'bqc,bchw->bqhw', outputs[-1], mask_feature)
        pred_feature = self.mlp(pred_feature)
        pred_mask = mask_feature + pred_feature
        pred_mask = self.fc(pred_mask)

        return pred_mask, mask_feature

    # def forward_prediction_head(self, output, mask_embed, spatial_shapes, level_start_index):
    #     masks = torch.einsum('bqc,bqn->bcn', output, mask_embed)
    #     splitted_masks = self.reform_output_squences(
    #         masks, spatial_shapes, level_start_index, 2)

    #     bs = output.shape[0]
    #     reforms = []
    #     for i, embed in enumerate(splitted_masks):
    #         embed = embed.view(
    #             bs, -1, spatial_shapes[i][0], spatial_shapes[i][1])
    #         reforms.append(embed)

    #     attn_mask = self.fpn(reforms)
    #     attn_mask = self.attn_fc(attn_mask)
    #     return attn_mask
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import build_generator
from ..utils.transformer import TransformerDecoder


class AVSegHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 embed_dim=256,
                 query_num=10, # 注意：在使用核引导时，实际 Query 数量是动态的，取决于 centroids 数量
                 num_layers=3,
                 num_heads=8,
                 multi_scale_indices=None,
                 query_generator=None,
                 transformer=None,
                 **kwargs):
        super(AVSegHead, self).__init__()

        # [新增检查] 确保 Transformer 的维度和 Head 的维度一致
        if transformer is not None and 'd_model' in transformer:
            if transformer['d_model'] != embed_dim:
                print(f"Warning: Config mismatch! Head embed_dim({embed_dim}) != Transformer d_model({transformer['d_model']})")
                # 甚至可以强制覆盖（可选）：
                # self.transformer_decoder.d_model = embed_dim

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

        # 构建 Query 生成器
        # Config 中需要配置 type='NucleiGuidedQueryGenerator'
        self.query_generator = build_generator(**query_generator)
        
        # 构建 Transformer 解码器
        self.transformer_decoder = TransformerDecoder(**transformer)

        # 类别预测层 (用于预测 Objectness / 存在概率)
        self.class_embed = nn.Linear(embed_dim, 1) # 二分类：是细胞 vs 背景

        # Mask 嵌入预测层
        self.mask_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 视觉特征投影层 (将 Backbone 输出通道映射到 embed_dim)
        self.input_proj = nn.ModuleList([
            nn.Conv2d(cur_in_channels, embed_dim, kernel_size=1)
            for cur_in_channels in selected_in_channels
        ])
        self.mask_feature_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(selected_in_channels), embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, x, omics_context, centroids=None):
        """
        Args:
            x: [Batch, Channel, H, W] 视觉特征图
            omics_context: [Batch, 1, Embed_Dim] 组学上下文特征 (替代原 audio)
            centroids: [Batch, N_cells, 2] 细胞质心坐标 (由 Module 3 使用)
        """
        # --- 1. 视觉特征处理 ---
        # 如果输入是多层特征列表，通常取最后一层
        if isinstance(x, list):
            visual_feats = [x[idx] for idx in self.multi_scale_indices]
        else:
            visual_feats = [x]

        projected_feats = []
        memory_tokens = []
        for feat, proj in zip(visual_feats, self.input_proj):
            cur_feat = proj(feat)
            projected_feats.append(cur_feat)
            memory_tokens.append(cur_feat.flatten(2).transpose(1, 2))

        fuse_h, fuse_w = projected_feats[0].shape[-2:]
        resized_feats = []
        for feat in projected_feats:
            if feat.shape[-2:] != (fuse_h, fuse_w):
                feat = F.interpolate(feat, size=(fuse_h, fuse_w), mode='bilinear', align_corners=False)
            resized_feats.append(feat)
        src = self.mask_feature_fuse(torch.cat(resized_feats, dim=1))
        src_flatten = torch.cat(memory_tokens, dim=1)

        # --- 2. 生成 Query (核心修改) ---
        if centroids is not None:
            # [Module 3] 使用质心坐标生成 Query -> [Batch, N_cells, Embed_Dim]
            query_embed = self.query_generator(centroids)
        else:
            # 防御性代码：如果没有传质心 (例如测试阶段如果用其他方式)，尝试回退
            # 但 NucleiGuidedQueryGenerator 可能不支持 tensor 输入，这里需注意
            # 假设为了兼容旧 Config，保留原逻辑
            query_embed = self.query_generator(omics_context)

        # --- 3. Transformer 解码 ---
        # query_embed: [Batch, N_cells, Dim] -> 找什么 (细胞)
        # src_flatten: [Batch, H*W, Dim]     -> 在哪里找 (图像)
        # omics_context: [Batch, 1, Dim]     -> 辅助信息 (组学特征作为 KV)
        
        # Decoder 输出: [Batch, N_cells, Dim]
        out = self.transformer_decoder(query_embed, src_flatten, omics_context)

        # --- 4. 生成预测结果 ---
        # (1) 预测分类/存在分数: [Batch, N_cells, 1]
        outputs_class = self.class_embed(out)

        # (2) 预测 Mask 权重: [Batch, N_cells, Dim]
        mask_embed = self.mask_embed(out)

        # (3) 生成最终 Mask: (Mask_Embed @ Visual_Features^T)
        # [B, N, C] @ [B, C, H*W] -> [B, N, H*W] -> reshape -> [B, N, H, W]
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, src)

        return outputs_class, outputs_mask
