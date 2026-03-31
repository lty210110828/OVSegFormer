import torch
import torch.nn as nn


class NucleiGuidedQueryGenerator(nn.Module):
    """
    核引导 Query 生成器 — OVSegFormer 的核心创新之一

    将细胞核质心的归一化坐标 (x, y) 通过 MLP 映射为高维 query embedding，
    替代原 AVSegFormer 中基于音频特征的 query 生成方式

    每个细胞核质心 → 一个 query → 用于 Transformer 解码器预测该细胞的分割掩膜

    设计思路:
        - 低维坐标 (2D) 通过 MLP 提升到高维语义空间 (256D)
        - LayerNorm 保证训练稳定性
        - Xavier 初始化确保初始阶段梯度均匀
    """
    def __init__(self, input_dim=2, embed_dim=256, hidden_dim=128):
        """
        Args:
            input_dim: 输入坐标维度 (2 = x, y)
            embed_dim: 输出 query 维度 (须与 Transformer d_model 一致)
            hidden_dim: MLP 隐藏层维度
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, centroids):
        """
        Args:
            centroids: [B, N_cells, 2] 归一化质心坐标

        Returns:
            [B, N_cells, embed_dim] query embeddings
        """
        return self.mlp(centroids)


# --- [以下是原 AVSegFormer 代码，保留以防报错，不注释] ---

class RepeatGenerator(nn.Module):
    def __init__(self, query_num) -> None:
        super().__init__()
        self.query_num = query_num

    def forward(self, audio_feat):
        return audio_feat.repeat(1, self.query_num, 1)


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, bias=False, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, bias=False, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, query, audio_feat):
        out1 = self.self_attn(query, query, query)[0]
        query = self.norm1(query+out1)
        out2 = self.cross_attn(query, audio_feat, audio_feat)[0]
        query = self.norm2(query+out2)
        out3 = self.ffn(query)
        query = self.norm3(query+out3)
        return query


class AttentionGenerator(nn.Module):
    def __init__(self, num_layers, query_num, embed_dim=256, num_heads=8, hidden_dim=1024):
        super().__init__()
        self.num_layers = num_layers
        self.query_num = query_num
        self.embed_dim = embed_dim
        self.query = nn.Embedding(query_num, embed_dim)
        self.layers = nn.ModuleList(
            [AttentionLayer(embed_dim, num_heads, hidden_dim)
             for i in range(num_layers)]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, audio_feat):
        bs = audio_feat.shape[0]
        query = self.query.weight[None, :, :].repeat(bs, 1, 1)
        for layer in self.layers:
            query = layer(query, audio_feat)
        return query


def build_generator(type, **kwargs):
    """
    工厂函数: 根据配置中的 type 字段构建对应的 Query 生成器
    支持: NucleiGuidedQueryGenerator (本项目), AttentionGenerator, RepeatGenerator (原项目)
    """
    if type == 'AttentionGenerator':
        return AttentionGenerator(**kwargs)
    elif type == 'RepeatGenerator':
        return RepeatGenerator(**kwargs)
    elif type == 'NucleiGuidedQueryGenerator':
        return NucleiGuidedQueryGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator type: {type}")
