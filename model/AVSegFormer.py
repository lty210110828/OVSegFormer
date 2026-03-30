'''
import torch
import torch.nn as nn
from .backbone import build_backbone
# from .neck import build_neck
from .head import build_head
from .vggish import VGGish


class AVSegFormer(nn.Module):
    def __init__(self,
                 backbone,
                 vggish,
                 head,
                 neck=None,
                 audio_dim=128,
                 embed_dim=256,
                 T=5,
                 freeze_audio_backbone=True,
                 *args, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.T = T
        self.freeze_audio_backbone = freeze_audio_backbone
        self.backbone = build_backbone(**backbone)
        self.vggish = VGGish(**vggish)
        self.head = build_head(**head)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)

        if self.freeze_audio_backbone:
            for p in self.vggish.parameters():
                p.requires_grad = False
        self.freeze_backbone(True)

        self.neck = neck
        # if neck is not None:
        #     self.neck = build_neck(**neck)
        # else:
        #     self.neck = None

    def freeze_backbone(self, freeze=False):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def mul_temporal_mask(self, feats, vid_temporal_mask_flag=None):
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
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def forward(self, audio, frames, vid_temporal_mask_flag=None):
        if vid_temporal_mask_flag is not None:
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(-1, 1, 1, 1)
        with torch.no_grad():
            audio_feat = self.vggish(audio)  # [B*T,128]

        audio_feat = audio_feat.unsqueeze(1)
        audio_feat = self.audio_proj(audio_feat)
        img_feat = self.extract_feat(frames)
        img_feat = self.mul_temporal_mask(img_feat, vid_temporal_mask_flag)

        pred, mask_feature = self.head(img_feat, audio_feat)
        pred = self.mul_temporal_mask(pred, vid_temporal_mask_flag)
        mask_feature = self.mul_temporal_mask(
            mask_feature, vid_temporal_mask_flag)

        return pred, mask_feature
'''
import torch
import torch.nn as nn
from .backbone import build_backbone
from .head import build_head
from .backbone.gat_encoder import GATEncoder
from torch_geometric.nn import knn_graph

class AVSegFormer(nn.Module):
    def __init__(self,
                 backbone,
                 head,
                 neck=None,
                 embed_dim=256,
                 omics_token_num=8,
                 omics_token_heads=8,
                 gene_vocab_size=20000,
                 gene_embed_dim=32,
                 omics_scalar_dim=1,
                 # [修改] 移除了 vggish, audio_dim 等音频相关参数
                 *args, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.gene_embed_dim = gene_embed_dim
        self.omics_scalar_dim = omics_scalar_dim
        self.backbone = build_backbone(**backbone)
        
        # [修改] 核心替换：用 GATEncoder 替代 VGGish
        # in_channels=2 代表 (x, y) 坐标，如果您有基因特征，这里要改成对应的维度
        self.gene_embedding = nn.Embedding(gene_vocab_size, gene_embed_dim)
        self.omics_encoder = GATEncoder(in_channels=2 + gene_embed_dim + omics_scalar_dim, out_channels=embed_dim)
        self.omics_token_num = omics_token_num
        self.omics_token = nn.Parameter(torch.randn(1, omics_token_num, embed_dim))
        self.omics_token_attn = nn.MultiheadAttention(embed_dim, omics_token_heads, batch_first=True)

        # [修改] 移除音频投影层
        # self.audio_proj = nn.Linear(audio_dim, embed_dim) 

        self.head = build_head(**head)
        
        # [修改] 既然是新任务，建议解冻 Backbone 进行微调
        self.freeze_backbone(False)

        self.neck = neck
        # if neck is not None:
        #     self.neck = build_neck(**neck)
        # else:
        #     self.neck = None

    def freeze_backbone(self, freeze=False):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def mul_temporal_mask(self, feats, vid_temporal_mask_flag=None):
        """
        处理时序掩膜的辅助函数 (保留以兼容部分 Head 的调用)
        """
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
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def _align_batch_list(self, data_list, batch_size, name):
        if data_list is None:
            return None
        if len(data_list) == batch_size:
            return data_list
        raise RuntimeError(f'{name} batch size mismatch: got {len(data_list)}, expected {batch_size}')

    def forward(self, imgs, omics_x, centroids, omics_gene_ids=None, omics_qv=None, vid_temporal_mask_flag=None):
        """
        Args:
            imgs: [B, 3, H, W] 图像切片
            omics_x: List of Tensor [N_genes, 2] (或者 Tensor [B, N, 2]) 转录本坐标
            centroids: [B, N_cells, 2] 细胞质心坐标
            vid_temporal_mask_flag: 保留接口，通常为 None
        """
        batch_size = imgs.shape[0]
        omics_x = self._align_batch_list(omics_x, batch_size, 'omics_x')
        omics_gene_ids = self._align_batch_list(omics_gene_ids, batch_size, 'omics_gene_ids')
        omics_qv = self._align_batch_list(omics_qv, batch_size, 'omics_qv')
        if vid_temporal_mask_flag is not None:
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(-1, 1, 1, 1)

        # 1. 提取视觉特征 (Module 2 Part A)
        img_feat = self.extract_feat(imgs) # Output: [B, C, H/32, W/32]
        img_feat = self.mul_temporal_mask(img_feat, vid_temporal_mask_flag)

        omics_feats_list = []

        if omics_gene_ids is None:
            omics_gene_ids = [torch.zeros((x.shape[0],), dtype=torch.long, device=imgs.device) for x in omics_x]
        if omics_qv is None:
            omics_qv = [torch.zeros((x.shape[0], self.omics_scalar_dim), dtype=imgs.dtype, device=imgs.device) for x in omics_x]

        for x, gene_ids, qv in zip(omics_x, omics_gene_ids, omics_qv):
            if x.device != imgs.device:
                x = x.to(imgs.device)
            if gene_ids.device != imgs.device:
                gene_ids = gene_ids.to(imgs.device)
            if qv.device != imgs.device:
                qv = qv.to(imgs.device)
            if x.shape[0] == 0:
                feat = torch.zeros((0, self.embed_dim), device=imgs.device, dtype=imgs.dtype)
                omics_feats_list.append(feat)
                continue
            if x.dtype != imgs.dtype:
                x = x.to(dtype=imgs.dtype)
            gene_ids = gene_ids.to(dtype=torch.long).clamp_min_(0)
            qv = qv.to(dtype=imgs.dtype)
            if qv.dim() == 1:
                qv = qv.unsqueeze(-1)
            gene_feat = self.gene_embedding(gene_ids)
            if gene_feat.dtype != imgs.dtype:
                gene_feat = gene_feat.to(dtype=imgs.dtype)
            node_feat = torch.cat([x, gene_feat, qv], dim=-1)
            k = min(10, x.shape[0] - 1)
            if k <= 0:
                feat = torch.zeros((x.shape[0], self.embed_dim), device=imgs.device, dtype=imgs.dtype)
            else:
                edge_index = knn_graph(x, k=k, loop=False).to(imgs.device)
                feat = self.omics_encoder(node_feat, edge_index)
            omics_feats_list.append(feat)

        omics_context_list = []
        for feat in omics_feats_list:
            if feat.shape[0] == 0:
                ctx = torch.zeros((1, self.omics_token_num, self.embed_dim), device=imgs.device, dtype=imgs.dtype)
            else:
                tokens = self.omics_token.expand(1, -1, -1).to(dtype=feat.dtype)
                ctx = self.omics_token_attn(tokens, feat.unsqueeze(0), feat.unsqueeze(0), need_weights=False)[0]
            omics_context_list.append(ctx.squeeze(0))
        omics_context = torch.stack(omics_context_list, dim=0)

        # 4. 解码预测 (Module 3)
        # 将 omics_context 作为 Key/Value 传入 (替代原来的 audio_feat)
        # 将 centroids 传入，供 NucleiGuidedQueryGenerator 生成 Query
        pred, mask_feature = self.head(img_feat, omics_context, centroids=centroids)
        
        pred = self.mul_temporal_mask(pred, vid_temporal_mask_flag)
        mask_feature = self.mul_temporal_mask(mask_feature, vid_temporal_mask_flag)

        return pred, mask_feature
