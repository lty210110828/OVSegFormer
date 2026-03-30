# 文件位置: model/backbone/gat_encoder.py
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GATEncoder(nn.Module):
    def __init__(self, in_channels=20, hidden_channels=64, out_channels=256, heads=4):
        """
        in_channels: 基因特征维度 (例如用了 gene embedding 或 simple one-hot)
        out_channels: 需要与视觉特征维度对齐 (通常是 256)
        """
        super().__init__()
        # 第一层 GAT: 聚合邻居信息
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        # 激活函数
        self.elu = nn.ELU()
        # 第二层 GAT: 输出最终特征
        # 输入维度是 hidden_channels * heads
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        """
        x: [Total_Nodes, In_Dim]
        edge_index: [2, Total_Edges]
        """
        x = self.conv1(x, edge_index)
        x = self.elu(x)
        x = self.conv2(x, edge_index)
        return x # 输出 [Total_Nodes, Out_Dim]