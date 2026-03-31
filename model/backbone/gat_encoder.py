import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATEncoder(nn.Module):
    """
    图注意力网络 (GATv2) 编码器 — 编码转录本级别的组学特征

    在 OVSegFormer 中用于将转录本的图结构数据 (坐标+基因嵌入+质量值) 编码为
    与视觉特征维度对齐的节点级表示

    典型调用场景:
        - AVSegFormer._encode_query_omics: 逐细胞构建图，取 anchor node 输出
        - AVSegFormer._encode_patch_omics: 全局 patch 图，mean pooling

    默认参数:
        in_channels = 2(坐标) + 32(基因嵌入) + 1(质量值) = 35
        out_channels = 256 (与 embed_dim 对齐)
    """
    def __init__(self, in_channels=20, hidden_channels=64, out_channels=256, heads=4):
        super().__init__()
        # 第一层 GAT: 多头注意力聚合邻居信息 → [N, hidden_channels * heads]
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.elu = nn.ELU()
        # 第二层 GAT: 单头输出最终节点特征 → [N, out_channels]
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N_nodes, in_channels] 节点特征矩阵
            edge_index: [2, N_edges] 边索引 (COO 格式)

        Returns:
            [N_nodes, out_channels] 编码后的节点特征
        """
        x = self.conv1(x, edge_index)
        x = self.elu(x)
        x = self.conv2(x, edge_index)
        return x
