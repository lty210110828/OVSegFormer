'''
[原 AVSegFormer 音频-视觉分割代码，保留但不注释]
'''
import torch
import torch.nn as nn
from .backbone import build_backbone
from .head import build_head
from .backbone.gat_encoder import GATEncoder
from torch_geometric.nn import knn_graph

class AVSegFormer(nn.Module):
    """
    OVSegFormer 主模型 — 空间转录组引导的细胞实例分割网络

    架构概览:
        1. ResNet50 视觉骨干 → 提取形态学图像的多尺度视觉特征
        2. 基因嵌入 + GAT 编码器 → 将转录本空间分布编码为组学特征
        3. AVSegHead → 核引导 Query + Transformer 解码器 → 预测每个细胞的分类分数与分割掩膜

    与原版 AVSegFormer 的核心区别:
        - 移除了音频分支 (VGGish)，替换为基因表达 (Gene Embedding + GAT) 分支
        - Query 不再由音频特征生成，而是由细胞核质心坐标通过 NucleiGuidedQueryGenerator 生成
        - 支持两种组学编码模式: _encode_query_omics (逐细胞图编码) 和 _encode_patch_omics (全局池化)
    """
    def __init__(self,
                 backbone,
                 head,
                 neck=None,
                 embed_dim=256,
                 gene_vocab_size=20000,
                 gene_embed_dim=32,
                 omics_scalar_dim=1,
                 query_graph_k=8,
                 *args, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.gene_embed_dim = gene_embed_dim
        self.omics_scalar_dim = omics_scalar_dim
        self.query_graph_k = int(query_graph_k)

        # 视觉骨干网络 (ResNet50)，输出多尺度特征图
        self.backbone = build_backbone(**backbone)

        # 基因词汇嵌入表: 将 codeword_index 映射为 gene_embed_dim 维向量
        self.gene_embedding = nn.Embedding(gene_vocab_size, gene_embed_dim)

        # GAT 图注意力编码器: 输入维度 = 坐标(2) + 基因嵌入(gene_embed_dim) + 质量值(omics_scalar_dim)
        self.omics_encoder = GATEncoder(
            in_channels=2 + gene_embed_dim + omics_scalar_dim,
            out_channels=embed_dim
        )

        # 分割头: 包含 Query 生成器 + Transformer 解码器 + 分类/掩膜预测层
        self.head = build_head(**head)

        # 默认不冻结骨干网络 (允许微调)
        self.freeze_backbone(False)

        self.neck = neck

    def freeze_backbone(self, freeze=False):
        """控制骨干网络参数是否参与梯度更新"""
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def mul_temporal_mask(self, feats, vid_temporal_mask_flag=None):
        """处理时序掩膜的辅助函数 (保留以兼容部分 Head 的调用)"""
        if vid_temporal_mask_flag is None:
            return feats
        else:
            if isinstance(feats, list):
                out = []
                for x in feats:
                    out.append(x * vid_temporal_mask_flag)
            elif isinstance(feats, torch.Tensor):
                out = feats * vid_temporal_mask_flag
            return out

    def extract_feat(self, x):
        """提取视觉特征: backbone → (可选) neck"""
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def _align_batch_list(self, data_list, batch_size, name):
        """校验 list 类型批次数据与期望 batch_size 是否一致"""
        if data_list is None:
            return None
        if len(data_list) == batch_size:
            return data_list
        raise RuntimeError(f'{name} batch size mismatch: got {len(data_list)}, expected {batch_size}')

    def _encode_query_omics(self, omics_x, centroids, omics_gene_ids=None, omics_qv=None, omics_valid_mask=None, query_valid_mask=None):
        """
        逐细胞构建图并编码组学特征 (tensorized batch 输入)

        对每个有效 query (细胞):
            1. 取出该细胞关联的转录本坐标、基因 ID、质量值
            2. 计算转录本相对于质心的偏移坐标
            3. 拼接 [偏移坐标, 基因嵌入, 质量值] 作为节点特征
            4. 构建星形图 (anchor node ↔ 转录本节点) + kNN 图 (转录本间)
            5. 通过 GAT 编码，取 anchor node 输出作为该细胞的组学表示

        Args:
            omics_x: [B, N_queries, N_tx, 2] 转录本归一化坐标
            centroids: [B, N_queries, 2] 质心归一化坐标
            omics_gene_ids: [B, N_queries, N_tx] 转录本基因 ID
            omics_qv: [B, N_queries, N_tx, 1] 转录本质量值
            omics_valid_mask: [B, N_queries, N_tx] 转录本有效掩膜
            query_valid_mask: [B, N_queries] query 有效掩膜

        Returns:
            query_omics: [B, N_queries, embed_dim] 每个细胞的组学特征向量
        """
        batch_size, max_queries, max_tx, _ = omics_x.shape
        device = centroids.device
        query_omics = torch.zeros((batch_size, max_queries, self.embed_dim), device=device, dtype=torch.float32)
        if omics_gene_ids is None:
            omics_gene_ids = torch.zeros((batch_size, max_queries, max_tx), dtype=torch.long, device=device)
        if omics_qv is None:
            omics_qv = torch.zeros((batch_size, max_queries, max_tx, self.omics_scalar_dim), dtype=torch.float32, device=device)
        if omics_valid_mask is None:
            omics_valid_mask = torch.ones((batch_size, max_queries, max_tx), dtype=torch.bool, device=device)
        if query_valid_mask is None:
            query_valid_mask = torch.ones((batch_size, max_queries), dtype=torch.bool, device=device)

        for batch_idx in range(batch_size):
            for query_idx in range(max_queries):
                # 跳过无效 query (padding 位置)
                if not bool(query_valid_mask[batch_idx, query_idx].item()):
                    continue
                tx_mask = omics_valid_mask[batch_idx, query_idx]
                # 跳过无转录本的 query
                if not bool(tx_mask.any().item()):
                    continue

                # 提取有效转录本数据并转移到目标设备
                tx_xy = omics_x[batch_idx, query_idx][tx_mask].to(device=device, dtype=torch.float32)
                tx_gene_ids = omics_gene_ids[batch_idx, query_idx][tx_mask].to(device=device, dtype=torch.long)
                tx_qv = omics_qv[batch_idx, query_idx][tx_mask].to(device=device, dtype=torch.float32)
                if tx_qv.dim() == 1:
                    tx_qv = tx_qv.unsqueeze(-1)  # [N_tx] → [N_tx, 1]

                centroid = centroids[batch_idx, query_idx].to(device=device, dtype=torch.float32)

                # 计算转录本相对于质心的偏移坐标
                rel_xy = tx_xy - centroid.unsqueeze(0)  # [N_tx, 2]

                # 构建节点特征: [偏移坐标(2), 基因嵌入(gene_embed_dim), 质量值(scalar_dim)]
                gene_feat = self.gene_embedding(tx_gene_ids).to(dtype=torch.float32)
                tx_feat = torch.cat([rel_xy, gene_feat, tx_qv], dim=-1)

                # 创建 anchor node (零向量) 作为该细胞的全局表示节点
                anchor_feat = torch.zeros((1, tx_feat.shape[-1]), dtype=torch.float32, device=device)
                node_feat = torch.cat([anchor_feat, tx_feat], dim=0)  # [1+N_tx, feat_dim]

                # 构建边: anchor↔所有转录本 (双向) + 转录本间 kNN
                tx_count = tx_feat.shape[0]
                transcript_nodes = torch.arange(1, tx_count + 1, device=device, dtype=torch.long)
                edge_parts = [
                    # anchor → 转录本
                    torch.stack([torch.zeros(tx_count, dtype=torch.long, device=device), transcript_nodes], dim=0),
                    # 转录本 → anchor
                    torch.stack([transcript_nodes, torch.zeros(tx_count, dtype=torch.long, device=device)], dim=0),
                ]
                if tx_count > 1:
                    k = min(self.query_graph_k, tx_count - 1)
                    tx_edge_index = knn_graph(rel_xy, k=k, loop=False)
                    # kNN 边索引偏移 +1 (因为 anchor 是 node 0)
                    edge_parts.append(tx_edge_index + 1)
                edge_index = torch.cat(edge_parts, dim=1)

                # GAT 编码，取 anchor node (index=0) 的输出作为细胞组学特征
                node_out = self.omics_encoder(node_feat, edge_index)
                query_omics[batch_idx, query_idx] = node_out[0]
        return query_omics

    def _encode_patch_omics(self, omics_x, centroids, omics_gene_ids=None, omics_qv=None, query_valid_mask=None):
        """
        全局 patch 级别的组学编码 (list 输入模式)

        对每个样本:
            1. 将 patch 内所有转录本构建为一个图
            2. 通过 GAT 编码后 mean pooling 得到全局组学特征
            3. 广播到该样本所有有效 query

        注意: 此模式下所有 query 共享相同的组学上下文，适用于粗粒度场景
        """
        batch_size = centroids.shape[0]
        omics_x = self._align_batch_list(omics_x, batch_size, 'omics_x')
        omics_gene_ids = self._align_batch_list(omics_gene_ids, batch_size, 'omics_gene_ids')
        omics_qv = self._align_batch_list(omics_qv, batch_size, 'omics_qv')
        max_queries = centroids.shape[1]
        device = centroids.device
        query_omics = torch.zeros((batch_size, max_queries, self.embed_dim), device=device, dtype=torch.float32)
        if omics_gene_ids is None:
            omics_gene_ids = [torch.zeros((x.shape[0],), dtype=torch.long, device=device) for x in omics_x]
        if omics_qv is None:
            omics_qv = [torch.zeros((x.shape[0], self.omics_scalar_dim), dtype=torch.float32, device=device) for x in omics_x]
        for batch_idx, (x, gene_ids, qv) in enumerate(zip(omics_x, omics_gene_ids, omics_qv)):
            if x.device != device:
                x = x.to(device)
            if gene_ids.device != device:
                gene_ids = gene_ids.to(device)
            if qv.device != device:
                qv = qv.to(device)
            if x.shape[0] == 0:
                continue
            x = x.to(dtype=torch.float32)
            qv = qv.to(dtype=torch.float32)
            if qv.dim() == 1:
                qv = qv.unsqueeze(-1)

            # 节点特征: [坐标(2), 基因嵌入, 质量值]
            gene_feat = self.gene_embedding(gene_ids.to(dtype=torch.long).clamp_min_(0)).to(dtype=torch.float32)
            node_feat = torch.cat([x, gene_feat, qv], dim=-1)

            if x.shape[0] > 1:
                # kNN 建图 + GAT 编码 + mean pooling
                edge_index = knn_graph(x, k=min(self.query_graph_k, x.shape[0] - 1), loop=False).to(device)
                node_out = self.omics_encoder(node_feat, edge_index)
                pooled = node_out.mean(dim=0)
            else:
                pooled = torch.zeros((self.embed_dim,), device=device, dtype=torch.float32)

            # 广播到所有有效 query
            valid_queries = query_valid_mask[batch_idx] if query_valid_mask is not None else torch.ones((max_queries,), dtype=torch.bool, device=device)
            query_omics[batch_idx, valid_queries] = pooled.unsqueeze(0)
        return query_omics

    def forward(self, imgs, omics_x, centroids, omics_gene_ids=None, omics_qv=None, omics_valid_mask=None, query_valid_mask=None, vid_temporal_mask_flag=None):
        """
        前向传播

        Args:
            imgs: [B, 3, H, W] 形态学图像
            omics_x: 转录本坐标 (tensor 时 [B,N,Q,2], list 时为 list of [N,2])
            centroids: [B, N_queries, 2] 细胞质心归一化坐标
            omics_gene_ids: 转录本基因 ID
            omics_qv: 转录本质量值
            omics_valid_mask: 转录本有效掩膜
            query_valid_mask: [B, N_queries] query 有效掩膜
            vid_temporal_mask_flag: 兼容参数

        Returns:
            outputs_class: [B, N_queries, 1] 每个细胞的 objectness 分数 (logits)
            outputs_mask: [B, N_queries, H, W] 每个细胞的分割掩膜 (logits)
        """
        batch_size = imgs.shape[0]
        if vid_temporal_mask_flag is not None:
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(-1, 1, 1, 1)

        # Step 1: 提取视觉特征
        img_feat = self.extract_feat(imgs)
        img_feat = self.mul_temporal_mask(img_feat, vid_temporal_mask_flag)

        # Step 2: 编码组学特征 (自动判断 tensor/list 输入模式)
        if torch.is_tensor(omics_x):
            # tensor 模式: 逐细胞图编码 (_encode_query_omics)
            if omics_x.device != imgs.device:
                omics_x = omics_x.to(imgs.device)
            if omics_gene_ids is not None and omics_gene_ids.device != imgs.device:
                omics_gene_ids = omics_gene_ids.to(imgs.device)
            if omics_qv is not None and omics_qv.device != imgs.device:
                omics_qv = omics_qv.to(imgs.device)
            if omics_valid_mask is not None and omics_valid_mask.device != imgs.device:
                omics_valid_mask = omics_valid_mask.to(imgs.device)
            if query_valid_mask is not None and query_valid_mask.device != imgs.device:
                query_valid_mask = query_valid_mask.to(imgs.device)
            query_omics = self._encode_query_omics(
                omics_x,
                centroids,
                omics_gene_ids=omics_gene_ids,
                omics_qv=omics_qv,
                omics_valid_mask=omics_valid_mask,
                query_valid_mask=query_valid_mask,
            ).to(dtype=imgs.dtype)
        else:
            # list 模式: 全局 patch 编码 (_encode_patch_omics)
            query_omics = self._encode_patch_omics(
                omics_x,
                centroids,
                omics_gene_ids=omics_gene_ids,
                omics_qv=omics_qv,
                query_valid_mask=query_valid_mask,
            ).to(dtype=imgs.dtype)

        # Step 3: 分割头预测 — 生成分类分数和实例掩膜
        pred, mask_feature = self.head(img_feat, query_omics, centroids=centroids, query_valid_mask=query_valid_mask)
        
        pred = self.mul_temporal_mask(pred, vid_temporal_mask_flag)
        mask_feature = self.mul_temporal_mask(mask_feature, vid_temporal_mask_flag)

        return pred, mask_feature
