'''
[原 OVSegDataset 简化版代码，保留但不注释]
'''
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import tifffile
from torchvision import transforms
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


class OVSegDataset(Dataset):
    """
    OVSegFormer 空间转录组数据集 — 加载 Xenium 平台数据用于细胞实例分割训练

    数据源 (data_root 目录下):
        - morphology.ome.tif: 形态学图像 (DAPI 荧光)
        - transcripts.parquet: 转录本坐标、基因 ID、质量值 (列: x_location, y_location, codeword_index, qv, is_gene)
        - cells.parquet: 细胞质心坐标 (列: cell_id, x_centroid, y_centroid)
        - cell_boundaries.parquet: 细胞边界多边形顶点 (列: cell_id, vertex_x, vertex_y)

    数据处理流水线:
        1. 初始化时加载所有元数据到内存 (Parquet 高效格式)
        2. 按 spatial_x 模式将细胞划分为 train/val/test (空间不重叠)
        3. __getitem__ 随机/确定性选取裁剪中心 → 读取图像 patch
        4. 根据 cell_boundaries 绘制每个细胞的实例掩膜
        5. 筛选 patch 内转录本，通过多边形命中检测分配到对应细胞
        6. 归一化所有坐标到 [0, 1]

    输出字段:
        img: [3, H, W] 形态学图像 (ImageNet 归一化)
        omics_x: [N_tx, 2] patch 内转录本归一化坐标
        omics_gene_ids: [N_tx] 转录本基因 ID (codeword_index)
        omics_qv: [N_tx, 1] 转录本质量值 (归一化到 [0, 1])
        query_tx_x: list of [N_tx_per_cell, 2] 每个细胞的关联转录本坐标
        query_tx_gene_ids: list of [N_tx_per_cell] 每个细胞的关联转录本基因 ID
        query_tx_qv: list of [N_tx_per_cell, 1] 每个细胞的关联转录本质量值
        centroids: [N_cells, 2] 细胞质心归一化坐标
        instance_masks: [N_cells, H, W] 每个细胞的二值实例掩膜
        instance_target_valid: [N_cells] 每个细胞是否有有效掩膜
        mask: [1, H, W] 语义分割 GT (所有细胞的合并掩膜)
    """
    def __init__(
        self,
        data_root,
        split='train',
        crop_size=512,
        pixel_size=0.2125,
        show_progress=True,
        num_samples=None,
        max_transcripts=2000,
        benchmark_seed=123,
        deterministic_eval=True,
        split_ratios=(0.8, 0.1, 0.1),
        enforce_disjoint_cells=True,
        split_mode='spatial_x',
        split_center_margin_um=None,
        tx_assign_mode='polygon',
        max_query_transcripts=64,
    ):
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.crop_size = crop_size
        self.pixel_size = pixel_size  # Xenium 分辨率 µm/pixel
        self.show_progress = show_progress
        self.max_transcripts = int(max_transcripts)
        self.benchmark_seed = int(benchmark_seed)
        self.deterministic_eval = bool(deterministic_eval)
        self.split_ratios = tuple(split_ratios)
        self.enforce_disjoint_cells = bool(enforce_disjoint_cells)
        self.split_mode = str(split_mode)
        self.split_center_margin_um = split_center_margin_um
        self.tx_assign_mode = str(tx_assign_mode)
        self.max_query_transcripts = int(max_query_transcripts)

        # ========== 第一阶段: 加载元数据 ==========
        print(f"正在加载元数据: {data_root}...")
        init_bar = None
        if self.show_progress and tqdm is not None:
            init_bar = tqdm(total=5, desc=f"OVSeg[{self.split}] 初始化", unit="step", dynamic_ncols=True)

        # 加载转录本数据 (仅保留 is_gene=True 的基因转录本)
        self.df_transcripts = pd.read_parquet(
            os.path.join(data_root, 'transcripts.parquet'),
            columns=['x_location', 'y_location', 'codeword_index', 'qv', 'is_gene']
        )
        if init_bar is not None:
            init_bar.update(1)

        # 加载细胞质心数据并按 cell_id 排序
        self.df_cells = pd.read_parquet(os.path.join(data_root, 'cells.parquet'))
        self.df_cells = self.df_cells.sort_values('cell_id').reset_index(drop=True)
        if init_bar is not None:
            init_bar.update(1)

        # 过滤仅保留基因转录本 (去除对照探针等)
        self.df_transcripts = self.df_transcripts[self.df_transcripts['is_gene']].reset_index(drop=True)

        # 提取为 numpy 数组加速后续索引
        self.cells_xy = self.df_cells[['x_centroid', 'y_centroid']].to_numpy(np.float32)
        self.cell_ids = self.df_cells['cell_id'].to_numpy()
        self.half_crop_um = float(self.crop_size * self.pixel_size / 2.0)

        # 构建数据划分 (空间不重叠)
        self.split_cell_mask, self.split_cell_indices, self.split_center_indices = self._build_split_membership()

        # 提取转录本数据为 numpy (释放 DataFrame 节省内存)
        self.tx_xy = self.df_transcripts[['x_location', 'y_location']].to_numpy(np.float32)
        self.tx_gene_ids = self.df_transcripts['codeword_index'].to_numpy(np.int64)
        # 质量值归一化: qv / 40 映射到 [0, 1]
        self.tx_qv = (self.df_transcripts['qv'].to_numpy(np.float32) / 40.0).clip(0.0, 1.0)
        self.gene_vocab_size = int(self.tx_gene_ids.max()) + 1 if self.tx_gene_ids.size > 0 else 1
        del self.df_transcripts

        # ========== 第二阶段: 加载图像 ==========
        self.tif_path = os.path.join(data_root, 'morphology.ome.tif')
        with tifffile.TiffFile(self.tif_path) as tif:
            self.img_shape = tif.pages[0].shape  # (H, W) 或 (C, H, W)
        if init_bar is not None:
            init_bar.update(1)

        # 尝试内存映射加速读取
        try:
            self.tif_memmap = tifffile.memmap(self.tif_path)
        except ValueError:
            self.tif_memmap = None
        if init_bar is not None:
            init_bar.update(1)
        self.tif_array_cache = None

        # 图像预处理: ToTensor + ImageNet 标准化
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )

        # 数据集长度 (每 epoch 采样 crop 数)
        self.length = int(num_samples) if num_samples is not None else (1000 if split == 'train' else 200)

        # ========== 第三阶段: 加载细胞边界 ==========
        print("正在加载边界数据...")
        self.df_boundaries = pd.read_parquet(os.path.join(data_root, 'cell_boundaries.parquet'))
        grouped = self.df_boundaries.groupby('cell_id')
        boundary_iter = grouped
        if self.show_progress and tqdm is not None:
            boundary_iter = tqdm(grouped, total=grouped.ngroups, desc=f"OVSeg[{self.split}] 边界索引", unit="cell", dynamic_ncols=True)

        # 构建 cell_id → 顶点坐标数组的字典，加速 __getitem__ 中的多边形绘制
        self.boundary_dict = {}
        for cell_id, group in boundary_iter:
            self.boundary_dict[cell_id] = group[['vertex_x', 'vertex_y']].to_numpy(np.float32)
        if init_bar is not None:
            init_bar.update(1)
            init_bar.close()

        # 验证/测试集: 预计算固定的裁剪中心索引 (确保可复现)
        self.fixed_cell_indices = None
        if self.split in ('val', 'test') and self.deterministic_eval:
            self.fixed_cell_indices = self._build_fixed_cell_indices()

    def __len__(self):
        return self.length

    def _build_fixed_cell_indices(self):
        """
        为验证/测试集构建固定裁剪中心序列

        使用确定性随机种子确保每次运行验证集的裁剪位置相同，
        便于公平比较不同 epoch 的验证指标
        """
        num_cells = self.split_center_indices.shape[0]
        if num_cells == 0:
            raise ValueError('cells.parquet 中未找到可用细胞质心')
        seed_offset = 0 if self.split == 'val' else 10000
        rng = np.random.default_rng(self.benchmark_seed + seed_offset)
        if self.length <= num_cells:
            return self.split_center_indices[rng.permutation(num_cells)[:self.length]]
        repeats = int(np.ceil(self.length / num_cells))
        chunks = [self.split_center_indices[rng.permutation(num_cells)] for _ in range(repeats)]
        return np.concatenate(chunks, axis=0)[:self.length]

    def _build_split_membership(self):
        """
        构建数据划分: 将细胞划分为 train/val/test，确保空间不重叠

        划分策略 (split_mode='spatial_x'):
            - 按 X 坐标排序细胞
            - 前 80% 的细胞 → train, 中间 10% → val, 后 10% → test
            - 这样不同 split 的细胞在空间上完全分离，避免信息泄露

        center_indices: 进一步过滤掉靠近 split 边界的细胞作为裁剪中心，
                        确保裁剪的 patch 不会跨越 split 边界

        Returns:
            split_cell_mask: [N_cells] bool 数组，标记哪些细胞属于当前 split
            split_cell_indices: 当前 split 的细胞索引
            split_center_indices: 可用作裁剪中心的细胞索引 (去除边界附近的)
        """
        num_cells = self.cells_xy.shape[0]
        if num_cells == 0:
            raise ValueError('cells.parquet 中未找到可用细胞质心')

        if not self.enforce_disjoint_cells:
            all_mask = np.ones((num_cells,), dtype=bool)
            all_idx = np.nonzero(all_mask)[0]
            return all_mask, all_idx, all_idx

        ratios = np.asarray(self.split_ratios, dtype=np.float64)
        if ratios.shape[0] != 3 or np.any(ratios < 0):
            ratios = np.asarray([0.8, 0.1, 0.1], dtype=np.float64)
        ratio_sum = float(ratios.sum())
        if ratio_sum <= 0:
            ratios = np.asarray([0.8, 0.1, 0.1], dtype=np.float64)
            ratio_sum = float(ratios.sum())
        ratios = ratios / ratio_sum

        if self.split_mode == 'random_cell':
            rng = np.random.default_rng(self.benchmark_seed + 2048)
            ordered_indices = rng.permutation(num_cells)
            ordered_coord = None
        else:
            # 按 X (或 Y) 坐标排序，实现空间连续划分
            axis = 0 if self.split_mode == 'spatial_x' else 1
            ordered_indices = np.argsort(self.cells_xy[:, axis], kind='mergesort')
            ordered_coord = self.cells_xy[ordered_indices, axis]

        # 计算各 split 的边界索引
        train_end = int(round(num_cells * ratios[0]))
        val_end = int(round(num_cells * (ratios[0] + ratios[1])))
        train_end = min(max(train_end, 1), num_cells)
        val_end = min(max(val_end, train_end + 1), num_cells) if num_cells > 1 else num_cells

        split_to_indices = {
            'train': ordered_indices[:train_end],
            'val': ordered_indices[train_end:val_end],
            'test': ordered_indices[val_end:],
        }
        split_indices = split_to_indices.get(self.split, ordered_indices)
        if split_indices.shape[0] == 0:
            split_indices = ordered_indices

        split_mask = np.zeros((num_cells,), dtype=bool)
        split_mask[split_indices] = True

        # 过滤掉靠近 split 边界的细胞 (它们的 patch 可能跨越 split)
        center_indices = split_indices
        if ordered_coord is not None and split_indices.shape[0] > 0:
            axis = 0 if self.split_mode == 'spatial_x' else 1
            split_coords = self.cells_xy[split_indices, axis]
            lower = float(split_coords.min())
            upper = float(split_coords.max())
            margin = self.half_crop_um if self.split_center_margin_um is None else float(self.split_center_margin_um)
            center_mask = split_mask & (self.cells_xy[:, axis] >= lower + margin) & (self.cells_xy[:, axis] <= upper - margin)
            center_indices = np.nonzero(center_mask)[0]
            if center_indices.shape[0] == 0:
                center_indices = split_indices

        return split_mask, split_indices, center_indices

    def _get_crop_center(self, index):
        """获取裁剪中心坐标: 验证集用固定索引，训练集随机选取"""
        if self.fixed_cell_indices is not None:
            cell_idx = int(self.fixed_cell_indices[index])
            return self.cells_xy[cell_idx]
        rand_local_idx = np.random.randint(0, self.split_center_indices.shape[0])
        rand_idx = int(self.split_center_indices[rand_local_idx])
        return self.cells_xy[rand_idx]

    def _select_local_transcripts(self, local_tx_xy, local_tx_gene_ids, local_tx_qv):
        """
        从 patch 内转录本中采样，控制数量不超过 max_transcripts

        polygon 模式: 不采样 (后续按细胞分配)
        其他模式: 训练时随机采样，验证时均匀采样
        """
        if self.tx_assign_mode == 'polygon':
            return local_tx_xy, local_tx_gene_ids, local_tx_qv
        if local_tx_xy.shape[0] <= self.max_transcripts:
            return local_tx_xy, local_tx_gene_ids, local_tx_qv
        if self.fixed_cell_indices is None:
            sample_idx = np.random.choice(local_tx_xy.shape[0], self.max_transcripts, replace=False)
            return local_tx_xy[sample_idx], local_tx_gene_ids[sample_idx], local_tx_qv[sample_idx]
        sort_idx = np.lexsort((local_tx_xy[:, 1], local_tx_xy[:, 0]))
        ordered_tx_xy = local_tx_xy[sort_idx]
        ordered_tx_gene_ids = local_tx_gene_ids[sort_idx]
        ordered_tx_qv = local_tx_qv[sort_idx]
        sample_idx = np.linspace(0, ordered_tx_xy.shape[0] - 1, num=self.max_transcripts, dtype=np.int64)
        return ordered_tx_xy[sample_idx], ordered_tx_gene_ids[sample_idx], ordered_tx_qv[sample_idx]

    def _select_query_transcripts(self, tx_xy, tx_gene_ids, tx_qv, centroid_um):
        """
        为单个细胞选择最近的 N 个转录本 (优先保留高质量值)

        排序分数 = 距离² - ε * qv，距离近且质量高的转录本优先保留
        """
        if tx_xy.shape[0] <= self.max_query_transcripts:
            return tx_xy, tx_gene_ids, tx_qv
        rel = tx_xy - centroid_um[None, :]
        dist2 = np.sum(rel * rel, axis=1)
        score = dist2 - 1e-3 * tx_qv
        keep_idx = np.argsort(score, kind='mergesort')[:self.max_query_transcripts]
        return tx_xy[keep_idx], tx_gene_ids[keep_idx], tx_qv[keep_idx]

    def _assign_transcripts_to_cells(self, local_tx_xy, local_tx_gene_ids, local_tx_qv, valid_cell_ids, local_cells_xy, instance_masks_tensor, x_min_um, y_min_um, denom_x, denom_y):
        """
        将 patch 内转录本分配到对应细胞，并为每个细胞选择关联转录本

        分配方式 (tx_assign_mode='polygon'):
            - 将转录本坐标转换为像素坐标
            - 查询该像素在哪个细胞的实例掩膜中 → 分配到该细胞
            - 未命中任何掩膜的转录本不分配

        Returns:
            query_tx_x: list of [N_tx, 2] 每个细胞的归一化转录本坐标
            query_tx_gene_ids: list of [N_tx] 每个细胞的转录本基因 ID
            query_tx_qv: list of [N_tx, 1] 每个细胞的转录本质量值
        """
        query_tx_x = []
        query_tx_gene_ids = []
        query_tx_qv = []
        if local_cells_xy.shape[0] == 0:
            return query_tx_x, query_tx_gene_ids, query_tx_qv
        if local_tx_xy.shape[0] == 0:
            for _ in range(local_cells_xy.shape[0]):
                query_tx_x.append(torch.zeros((0, 2), dtype=torch.float32))
                query_tx_gene_ids.append(torch.zeros((0,), dtype=torch.long))
                query_tx_qv.append(torch.zeros((0, 1), dtype=torch.float32))
            return query_tx_x, query_tx_gene_ids, query_tx_qv

        # 计算每个转录本所属的细胞索引 (-1 = 未分配)
        owner = np.full((local_tx_xy.shape[0],), -1, dtype=np.int64)
        if self.tx_assign_mode == 'polygon':
            # 多边形命中检测: 转录本像素坐标 → 查实例掩膜
            tx_px_x = np.clip(np.floor((local_tx_xy[:, 0] - x_min_um) / self.pixel_size).astype(np.int32), 0, self.crop_size - 1)
            tx_px_y = np.clip(np.floor((local_tx_xy[:, 1] - y_min_um) / self.pixel_size).astype(np.int32), 0, self.crop_size - 1)
            mask_np = instance_masks_tensor.numpy()
            for cell_idx in range(mask_np.shape[0]):
                hits = mask_np[cell_idx, tx_px_y, tx_px_x] > 0
                assignable = hits & (owner < 0)
                owner[assignable] = cell_idx
        else:
            # 最近质心分配
            dist = np.sum((local_tx_xy[:, None, :] - local_cells_xy[None, :, :]) ** 2, axis=-1)
            owner = np.argmin(dist, axis=1).astype(np.int64)

        # 为每个细胞收集其转录本并归一化坐标
        for cell_idx in range(local_cells_xy.shape[0]):
            tx_idx = np.nonzero(owner == cell_idx)[0]
            if tx_idx.shape[0] == 0:
                query_tx_x.append(torch.zeros((0, 2), dtype=torch.float32))
                query_tx_gene_ids.append(torch.zeros((0,), dtype=torch.long))
                query_tx_qv.append(torch.zeros((0, 1), dtype=torch.float32))
                continue
            cell_tx_xy = local_tx_xy[tx_idx]
            cell_tx_gene_ids = local_tx_gene_ids[tx_idx]
            cell_tx_qv = local_tx_qv[tx_idx]
            # 选择最近的 N 个转录本
            cell_tx_xy, cell_tx_gene_ids, cell_tx_qv = self._select_query_transcripts(
                cell_tx_xy, cell_tx_gene_ids, cell_tx_qv,
                local_cells_xy[cell_idx],
            )
            # 坐标归一化到 [0, 1]
            x_norm = (cell_tx_xy[:, 0] - x_min_um) / denom_x
            y_norm = (cell_tx_xy[:, 1] - y_min_um) / denom_y
            query_tx_x.append(torch.from_numpy(np.stack([x_norm, y_norm], axis=1).astype(np.float32)))
            query_tx_gene_ids.append(torch.from_numpy(cell_tx_gene_ids.astype(np.int64)))
            query_tx_qv.append(torch.from_numpy(cell_tx_qv.astype(np.float32)).unsqueeze(-1))
        return query_tx_x, query_tx_gene_ids, query_tx_qv

    def __getitem__(self, index):
        """
        获取一个训练/验证样本

        流程:
            1. 选取裁剪中心 (训练随机/验证固定)
            2. 读取形态学图像 patch
            3. 绘制每个细胞的实例掩膜 (cell_boundaries → cv2.fillPoly)
            4. 筛选 patch 内转录本并分配到细胞
            5. 归一化所有坐标到 [0, 1]
        """
        # Step 1: 确定裁剪中心 (µm) → 计算像素级裁剪边界
        cx_micron, cy_micron = self._get_crop_center(index)

        cx_px = int(cx_micron / self.pixel_size)
        cy_px = int(cy_micron / self.pixel_size)
        half_size = self.crop_size // 2
        img_h, img_w = self.img_shape
        # 确保裁剪窗口不超出图像边界
        if img_w >= self.crop_size:
            x_min = max(0, min(cx_px - half_size, img_w - self.crop_size))
            x_max = x_min + self.crop_size
        else:
            x_min = 0
            x_max = img_w
        if img_h >= self.crop_size:
            y_min = max(0, min(cy_px - half_size, img_h - self.crop_size))
            y_max = y_min + self.crop_size
        else:
            y_min = 0
            y_max = img_h

        # 像素边界转换为物理坐标 (µm)
        x_min_um, x_max_um = x_min * self.pixel_size, x_max * self.pixel_size
        y_min_um, y_max_um = y_min * self.pixel_size, y_max * self.pixel_size

        # Step 2: 读取图像 patch (支持 memmap 加速)
        if self.tif_memmap is None:
            if self.tif_array_cache is None:
                with tifffile.TiffFile(self.tif_path) as tif:
                    self.tif_array_cache = tif.pages[0].asarray()
            if self.tif_array_cache.ndim == 2:
                img_patch = self.tif_array_cache[y_min:y_max, x_min:x_max]
            else:
                if self.tif_array_cache.shape[0] <= 4:
                    img_patch = np.moveaxis(self.tif_array_cache[:, y_min:y_max, x_min:x_max], 0, -1)
                else:
                    img_patch = self.tif_array_cache[y_min:y_max, x_min:x_max, :]
        elif self.tif_memmap.ndim == 2:
            img_patch = self.tif_memmap[y_min:y_max, x_min:x_max]
        else:
            if self.tif_memmap.shape[0] <= 4:
                img_patch = np.moveaxis(self.tif_memmap[:, y_min:y_max, x_min:x_max], 0, -1)
            else:
                img_patch = self.tif_memmap[y_min:y_max, x_min:x_max, :]

        # 边缘 padding: 如果 patch 不足 crop_size，补零
        if img_patch.shape[0] < self.crop_size or img_patch.shape[1] < self.crop_size:
            target_h = self.crop_size
            target_w = self.crop_size
            pad_y = target_h - img_patch.shape[0]
            pad_x = target_w - img_patch.shape[1]
            pad_y = max(0, pad_y)
            pad_x = max(0, pad_x)
            if img_patch.ndim == 2:
                pad_width = ((0, pad_y), (0, pad_x))
            else:
                pad_width = ((0, pad_y), (0, pad_x), (0, 0))
            img_patch = np.pad(img_patch, pad_width, mode='constant', constant_values=0)

        # uint16 → float32 归一化 + 单通道扩展为 3 通道 + ImageNet 标准化
        img_patch = img_patch.astype('float32') / 65535.0
        img_tensor = self.to_tensor(img_patch)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        img_tensor = self.normalize(img_tensor)

        # Step 3: 筛选当前 split 中属于此 patch 的细胞 → 绘制实例掩膜
        local_mask = (
            (self.cells_xy[:, 0] >= x_min_um) & (self.cells_xy[:, 0] < x_max_um) &
            (self.cells_xy[:, 1] >= y_min_um) & (self.cells_xy[:, 1] < y_max_um)
        ) & self.split_cell_mask
        local_idx = np.nonzero(local_mask)[0]
        valid_cell_ids = self.cell_ids[local_idx]
        local_cells_xy = self.cells_xy[local_idx]

        mask = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)  # 语义掩膜 (所有细胞合并)
        instance_masks = []       # 每个细胞的独立二值掩膜
        instance_target_valid = []  # 每个细胞是否有有效掩膜

        for cell_id in valid_cell_ids:
            boundary = self.boundary_dict.get(cell_id)
            if boundary is None:
                instance_masks.append(np.zeros((self.crop_size, self.crop_size), dtype=np.uint8))
                instance_target_valid.append(False)
                continue
            # 边界顶点从 µm 转换为 patch 内像素坐标
            pts = boundary / self.pixel_size
            pts[:, 0] -= x_min
            pts[:, 1] -= y_min
            pts = pts.astype(np.int32).reshape((-1, 1, 2))
            cell_mask = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)
            cv2.fillPoly(cell_mask, [pts], 1)
            if cell_mask.sum() == 0:
                instance_masks.append(cell_mask)
                instance_target_valid.append(False)
                continue
            cv2.fillPoly(mask, [pts], 1)
            instance_masks.append(cell_mask)
            instance_target_valid.append(True)

        mask_tensor = torch.from_numpy(mask).long().unsqueeze(0)  # [1, H, W]
        if len(instance_masks) > 0:
            instance_masks_tensor = torch.from_numpy(np.stack(instance_masks, axis=0)).float()  # [N_cells, H, W]
            instance_target_valid_tensor = torch.tensor(instance_target_valid, dtype=torch.bool)
        else:
            instance_masks_tensor = torch.zeros((0, self.crop_size, self.crop_size), dtype=torch.float32)
            instance_target_valid_tensor = torch.zeros((0,), dtype=torch.bool)

        # Step 4: 筛选 patch 内转录本 → 归一化坐标
        tx_mask = (
            (self.tx_xy[:, 0] >= x_min_um) & (self.tx_xy[:, 0] < x_max_um) &
            (self.tx_xy[:, 1] >= y_min_um) & (self.tx_xy[:, 1] < y_max_um)
        )
        local_tx_xy = self.tx_xy[tx_mask]
        local_tx_gene_ids = self.tx_gene_ids[tx_mask]
        local_tx_qv = self.tx_qv[tx_mask]
        local_tx_xy, local_tx_gene_ids, local_tx_qv = self._select_local_transcripts(
            local_tx_xy, local_tx_gene_ids, local_tx_qv,
        )
        denom_x = max(x_max_um - x_min_um, 1e-6)
        denom_y = max(y_max_um - y_min_um, 1e-6)
        if local_tx_xy.shape[0] > 0:
            x_norm = (local_tx_xy[:, 0] - x_min_um) / denom_x
            y_norm = (local_tx_xy[:, 1] - y_min_um) / denom_y
            x_feat = torch.from_numpy(np.stack([x_norm, y_norm], axis=1).astype(np.float32))
            x_gene_ids = torch.from_numpy(local_tx_gene_ids.astype(np.int64))
            x_qv = torch.from_numpy(local_tx_qv.astype(np.float32)).unsqueeze(-1)
        else:
            x_feat = torch.zeros((0, 2), dtype=torch.float32)
            x_gene_ids = torch.zeros((0,), dtype=torch.long)
            x_qv = torch.zeros((0, 1), dtype=torch.float32)

        # 将转录本分配到对应细胞
        query_tx_x, query_tx_gene_ids, query_tx_qv = self._assign_transcripts_to_cells(
            local_tx_xy, local_tx_gene_ids, local_tx_qv,
            valid_cell_ids, local_cells_xy, instance_masks_tensor,
            x_min_um, y_min_um, denom_x, denom_y,
        )

        # Step 5: 质心坐标归一化
        if local_cells_xy.shape[0] > 0:
            centroids = local_cells_xy.copy()
            centroids[:, 0] = (centroids[:, 0] - x_min_um) / denom_x
            centroids[:, 1] = (centroids[:, 1] - y_min_um) / denom_y
            centroids = torch.from_numpy(centroids.astype(np.float32))
        else:
            centroids = torch.zeros((0, 2), dtype=torch.float32)

        return {
            'img': img_tensor,
            'omics_x': x_feat,
            'omics_gene_ids': x_gene_ids,
            'omics_qv': x_qv,
            'query_tx_x': query_tx_x,
            'query_tx_gene_ids': query_tx_gene_ids,
            'query_tx_qv': query_tx_qv,
            'centroids': centroids,
            'instance_masks': instance_masks_tensor,
            'instance_target_valid': instance_target_valid_tensor,
            'mask': mask_tensor,
        }
