'''
[原 AVSegFormer Transformer 代码 (使用 MSDeformAttn)，保留但不注释]
'''
import torch
import torch.nn as nn


class CrossModalDecoderLayer(nn.Module):
    """
    跨模态 Transformer 解码层 — 核心特征融合单元

    每个 query (代表一个细胞) 通过以下三步完成信息聚合:
        1. Self-Attention: query 之间交换信息 (细胞间关系建模)
        2. Visual Cross-Attention: query 关注局部视觉 token (形态学特征)
        3. Omics Fusion: 组学上下文通过门控机制与视觉特征融合
        4. FFN: 前馈网络精炼

    门控融合机制:
        gate = σ(W[q; visual_ctx; omics_ctx])
        fused = gate * visual_ctx + (1-gate) * omics_ctx
    """
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()
        # Query 自注意力: 建模细胞间关系
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 视觉交叉注意力: query 关注视觉 memory tokens (可带局部掩膜)
        self.visual_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 组学交叉注意力: query 关注组学特征 (当维度不匹配时使用)
        self.omics_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # 门控融合: 根据 [query, visual, omics] 三者动态决定视觉/组学的权重
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid(),
        )
        # 投影融合后的特征回 d_model 维度
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, query, visual_feat, omics_feat, visual_attn_mask=None, visual_attn_bias=None):
        """
        Args:
            query: [B, N_queries, d_model] 当前 query 状态
            visual_feat: [B, N_tokens, d_model] 视觉 memory tokens
            omics_feat: [B, N_queries, d_model] 组学上下文特征
            visual_attn_mask: [B*nhead, N_queries, N_tokens] 局部视觉注意力掩膜

        Returns:
            [B, N_queries, d_model] 更新后的 query
        """
        # Step 1: Query 自注意力
        self_attended = self.self_attn(query, query, query)[0]
        query = self.norm1(query + self.dropout1(self_attended))

        visual_attn_prior = None
        if visual_attn_mask is not None and visual_attn_bias is not None:
            visual_attn_prior = visual_attn_mask + visual_attn_bias
        elif visual_attn_mask is not None:
            visual_attn_prior = visual_attn_mask
        elif visual_attn_bias is not None:
            visual_attn_prior = visual_attn_bias
        visual_ctx = self.visual_cross_attn(query, visual_feat, visual_feat, attn_mask=visual_attn_prior)[0]
        query_visual = self.norm2(query + self.dropout2(visual_ctx))

        # Step 3: 组学特征融合
        if omics_feat.shape[1] == query_visual.shape[1]:
            # 维度匹配时直接使用组学特征
            omics_ctx = omics_feat
        else:
            # 维度不匹配时通过交叉注意力对齐
            omics_ctx = self.omics_cross_attn(query_visual, omics_feat, omics_feat)[0]

        # 门控融合: 动态平衡视觉与组学信息
        gate = self.fusion_gate(torch.cat([query_visual, visual_ctx, omics_ctx], dim=-1))
        fused_ctx = gate * visual_ctx + (1.0 - gate) * omics_ctx
        fused_query = self.fusion_proj(torch.cat([query_visual, fused_ctx], dim=-1))
        query = self.norm3(query_visual + self.dropout3(fused_query))

        # Step 4: FFN
        ffn_out = self.linear2(self.dropout4(self.activation(self.linear1(query))))
        query = self.norm4(query + ffn_out)
        return query


class TransformerDecoder(nn.Module):
    """
    Transformer 解码器 — 堆叠多层 CrossModalDecoderLayer

    逐层精炼 query 表示，每层都重新与视觉/组学特征交互
    """
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=3, dim_feedforward=1024, dropout=0.1, activation="relu", **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_decoder_layers)
        ])

    def forward(self, query, visual_feat, omics_feat, visual_attn_mask=None, visual_attn_bias=None):
        """
        Args:
            query: [B, N_queries, d_model] 初始 query embeddings
            visual_feat: [B, N_tokens, d_model] 视觉 memory tokens
            omics_feat: [B, N_queries, d_model] 组学上下文特征
            visual_attn_mask: 局部视觉注意力掩膜

        Returns:
            [B, N_queries, d_model] 最终精炼后的 query 表示
        """
        out = query
        for layer in self.layers:
            out = layer(out, visual_feat, omics_feat, visual_attn_mask=visual_attn_mask, visual_attn_bias=visual_attn_bias)
        return out


def build_transformer(type, **kwargs):
    """工厂函数: 根据配置构建 Transformer 解码器"""
    return TransformerDecoder(**kwargs)
