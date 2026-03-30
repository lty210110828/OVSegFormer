import torch
import torch.nn as nn

# --- [新增核心模块] Module 3: 核引导 Query 生成器 ---
class NucleiGuidedQueryGenerator(nn.Module):
    def __init__(self, input_dim=2, embed_dim=256, hidden_dim=128):
        """
        Args:
            input_dim: 输入坐标维度，通常是 2 (x, y)
            embed_dim: 输出 Query 的维度，必须与 Transformer Decoder 的 hidden_dim 一致 (通常 256)
            hidden_dim: 中间层维度
        """
        super().__init__()
        # 使用 MLP (多层感知机) 将低维坐标映射为高维语义向量
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim) # LayerNorm 有助于训练稳定
        )

        # 权重初始化 (可选，但推荐)
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
            centroids: 归一化后的质心坐标 [Batch, N_cells, 2]
        Returns:
            query_embeddings: [Batch, N_cells, embed_dim]
        """
        # 简单直接：坐标 -> 向量
        return self.mlp(centroids)


# --- [以下是原代码，保留以防报错] ---

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


# --- [修改后的 Builder 函数] ---
def build_generator(type, **kwargs):
    if type == 'AttentionGenerator':
        return AttentionGenerator(**kwargs)
    elif type == 'RepeatGenerator':
        return RepeatGenerator(**kwargs)
    elif type == 'NucleiGuidedQueryGenerator': # <--- 新增：注册您的模块
        return NucleiGuidedQueryGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator type: {type}")