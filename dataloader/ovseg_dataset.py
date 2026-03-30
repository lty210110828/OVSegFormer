'''
# 文件位置: dataloader/ovseg_dataset.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2  # 需要导入 opencv
import tifffile
from torchvision import transforms
from torch_geometric.data import Data

class OVSegDataset(Dataset):
    def __init__(self, data_root, split='train', crop_size=512, pixel_size=0.2125):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.crop_size = crop_size
        self.pixel_size = pixel_size # Xenium 标准分辨率 um/pixel
        
        # 1. 预加载大文件元数据 (不要全读进内存，只读索引)
        print(f"正在加载元数据: {data_root}...")
        # 读取转录本 (建议用 parquet 加速)
        self.df_transcripts = pd.read_parquet(os.path.join(data_root, 'transcripts.parquet'))
        # 读取质心
        self.df_cells = pd.read_parquet(os.path.join(data_root, 'cells.parquet'))
        
        # 打开图像句柄 (懒加载)
        self.tif_path = os.path.join(data_root, 'morphology.ome.tif')
        with tifffile.TiffFile(self.tif_path) as tif:
            self.img_shape = tif.pages[0].shape # (H, W)
        
        # 定义图像增强
        self.img_transform = transforms.Compose([
            transforms.ToTensor(), # 转为 [C, H, W] 并归一化到 0-1
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # 模拟数据长度 (因为是随机切片，我们可以定义一个 epoch 切多少张)
        self.length = 1000 if split == 'train' else 200

        # [新增] 读取细胞边界文件 (用于生成 Mask)
        print("正在加载边界数据...")
        self.df_boundaries = pd.read_parquet(os.path.join(data_root, 'cell_boundaries.parquet'))
        
        # 建立 cell_id 到边界多边形的索引，加速读取 (这步可能有点慢，建议只做一次或存成 cache)
        # 这里为了简单展示逻辑，假设 df_boundaries 有 'cell_id', 'vertex_x', 'vertex_y'
        # 实际 Xenium 数据通常每个 cell_id 对应多行(多个顶点)
        self.grouped_boundaries = self.df_boundaries.groupby('cell_id')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # --- 1. 随机选择切片中心 ---
        # 简单策略：随机选一个有细胞的位置作为中心，避免切到空白背景
        rand_cell = self.df_cells.sample(1).iloc[0]
        cx_micron, cy_micron = rand_cell['x_centroid'], rand_cell['y_centroid']
        
        # 转换物理坐标(um) -> 像素坐标(px)
        cx_px = int(cx_micron / self.pixel_size)
        cy_px = int(cy_micron / self.pixel_size)
        
        # 计算切片边界
        half_size = self.crop_size // 2
        x_min = max(0, cx_px - half_size)
        y_min = max(0, cy_px - half_size)
        x_max = min(self.img_shape[1], cx_px + half_size)
        y_max = min(self.img_shape[0], cy_px + half_size)
        
        # --- 2. 读取图像 Patch ---
        # 使用 tifffile 读取特定区域 (高效)
        with tifffile.TiffFile(self.tif_path) as tif:
            # 注意: tifffile 读取顺序通常是 [Y, X]
            img_patch = tif.pages[0].asarray()[y_min:y_max, x_min:x_max]
            
        # [修复] 手动转换为 float32 并归一化，解决 uint16 报错
        img_patch = img_patch.astype('float32') / 65535.0    
        # 如果切出来的图不够大 (在边缘)，需要 Padding (这里省略，假设都够大)
        # 转为 Tensor
        img_tensor = self.img_transform(img_patch) # [3, 512, 512] (假设 RGB)
        # 如果是单通道 DAPI，需要 repeat 成 3 通道:
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)

        # --- 3. [修改] 读取边界并生成 GT Mask ---
        # 找出这个区域内的所有 cell_id
        local_cells = self.df_cells[
            (self.df_cells['x_centroid'] >= x_min_um) & (self.df_cells['x_centroid'] < x_max_um) &
            (self.df_cells['y_centroid'] >= y_min_um) & (self.df_cells['y_centroid'] < y_max_um)
        ]
        valid_cell_ids = local_cells['cell_id'].values

        # 创建一个全黑的 Mask [H, W]
        mask = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)

        for cell_id in valid_cell_ids:
            try:
                # 获取该细胞的所有顶点
                cell_boundary = self.grouped_boundaries.get_group(cell_id)
                # 转换微米 -> 像素 (相对于 Patch 左上角)
                poly_pts = []
                for _, row in cell_boundary.iterrows():
                    px = int(row['vertex_x'] / self.pixel_size) - x_min
                    py = int(row['vertex_y'] / self.pixel_size) - y_min
                    poly_pts.append([px, py])
                
                # 用 OpenCV 画实心多边形
                pts = np.array(poly_pts, np.int32)
                pts = pts.reshape((-1, 1, 2))
                # 填充白色 (1)
                cv2.fillPoly(mask, [pts], 1) 
            except KeyError:
                continue # 有些细胞可能没边界数据，跳过

        # 转为 Tensor [1, H, W]
        mask_tensor = torch.from_numpy(mask).long().unsqueeze(0)

        # --- 4. 切分转录本 (Omics Data) ---
        # 转换边界回微米，用于筛选 dataframe
        x_min_um, x_max_um = x_min * self.pixel_size, x_max * self.pixel_size
        y_min_um, y_max_um = y_min * self.pixel_size, y_max * self.pixel_size
        
        # 筛选落在此区域的转录本
        local_tx = self.df_transcripts[
            (self.df_transcripts['x_location'] >= x_min_um) & 
            (self.df_transcripts['x_location'] < x_max_um) &
            (self.df_transcripts['y_location'] >= y_min_um) & 
            (self.df_transcripts['y_location'] < y_max_um)
        ].copy() # 必须 copy 以免 SettingWithCopyWarning
        
        # 归一化转录本坐标到 [0, 1] (相对于 Patch)
        local_tx['x_norm'] = (local_tx['x_location'] - x_min_um) / (x_max_um - x_min_um)
        local_tx['y_norm'] = (local_tx['y_location'] - y_min_um) / (y_max_um - y_min_um)
        
        # 构建图数据 (这里简化，假设取前 1000 个点，防止爆显存)
        if len(local_tx) > 2000:
            local_tx = local_tx.sample(2000)
            
        # Node Features: [N, 2] (仅使用坐标，如果有 gene embedding 可以加上)
        x_feat = torch.tensor(local_tx[['x_norm', 'y_norm']].values, dtype=torch.float)
        
        # 简单的 k-NN 建图 (需要在 forward 里动态建，或者这里建好 edge_index)
        # 这里为了 Batch 方便，我们通常只返回节点特征，在模型里用 knn_graph 建图
        
        # --- 5. 切分质心 (Module 3 Query) ---
        local_cells = self.df_cells[
            (self.df_cells['x_centroid'] >= x_min_um) & (self.df_cells['x_centroid'] < x_max_um) &
            (self.df_cells['y_centroid'] >= y_min_um) & (self.df_cells['y_centroid'] < y_max_um)
        ]
        
        centroids = torch.tensor(local_cells[['x_centroid', 'y_centroid']].values, dtype=torch.float)
        # 同样归一化到 [0, 1]
        centroids[:, 0] = (centroids[:, 0] - x_min_um) / (x_max_um - x_min_um)
        centroids[:, 1] = (centroids[:, 1] - y_min_um) / (y_max_um - y_min_um)

        # 返回字典
        return {
            'img': img_tensor,           
            'omics_x': x_feat,           
            'centroids': centroids,      
            'mask': mask_tensor, # <--- 现在这里有了真实的 Mask
            # 还可以加一个 'vid_temporal_mask_flag': torch.tensor([1]) 兼容原代码接口
        }
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
    """OvSeg数据集加载类，用于处理Xenium空间转录组数据
    
    该数据集类负责加载和预处理空间转录组数据，包括:
    1. 形态学图像(TIFF格式)
    2. 转录本数据
    3. 细胞质心数据
    4. 细胞边界数据
    
    Args:
        data_root (str): 数据根目录路径
        split (str, optional): 数据集类型('train'或'test')，默认为'train'
        crop_size (int, optional): 图像切片大小，默认为512
        pixel_size (float, optional): 像素分辨率(微米/像素)，默认为0.2125(Xenium标准)
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
    ):
        # 初始化父类Dataset
        super().__init__()
        
        # 保存参数到实例变量
        self.data_root = data_root        # 数据根目录
        self.split = split                # 数据集类型('train'/'test')
        self.crop_size = crop_size        # 图像切片尺寸
        self.pixel_size = pixel_size      # 像素分辨率(um/px)
        self.show_progress = show_progress
        self.max_transcripts = int(max_transcripts)
        self.benchmark_seed = int(benchmark_seed)
        self.deterministic_eval = bool(deterministic_eval)
        self.split_ratios = tuple(split_ratios)
        self.enforce_disjoint_cells = bool(enforce_disjoint_cells)
        self.split_mode = str(split_mode)
        self.split_center_margin_um = split_center_margin_um
        
        # 加载元数据
        print(f"正在加载元数据: {data_root}...")
        init_bar = None
        if self.show_progress and tqdm is not None:
            init_bar = tqdm(total=5, desc=f"OVSeg[{self.split}] 初始化", unit="step", dynamic_ncols=True)
        # 加载转录本数据(Parquet格式，高效存储)
        self.df_transcripts = pd.read_parquet(
            os.path.join(data_root, 'transcripts.parquet'),
            columns=['x_location', 'y_location', 'codeword_index', 'qv', 'is_gene']
        )
        if init_bar is not None:
            init_bar.update(1)
        self.df_cells = pd.read_parquet(os.path.join(data_root, 'cells.parquet'))
        self.df_cells = self.df_cells.sort_values('cell_id').reset_index(drop=True)
        if init_bar is not None:
            init_bar.update(1)
        self.df_transcripts = self.df_transcripts[self.df_transcripts['is_gene']].reset_index(drop=True)
        self.cells_xy = self.df_cells[['x_centroid', 'y_centroid']].to_numpy(np.float32)
        self.cell_ids = self.df_cells['cell_id'].to_numpy()
        self.half_crop_um = float(self.crop_size * self.pixel_size / 2.0)
        self.split_cell_mask, self.split_cell_indices, self.split_center_indices = self._build_split_membership()
        self.tx_xy = self.df_transcripts[['x_location', 'y_location']].to_numpy(np.float32)
        self.tx_gene_ids = self.df_transcripts['codeword_index'].to_numpy(np.int64)
        self.tx_qv = (self.df_transcripts['qv'].to_numpy(np.float32) / 40.0).clip(0.0, 1.0)
        self.gene_vocab_size = int(self.tx_gene_ids.max()) + 1 if self.tx_gene_ids.size > 0 else 1
        del self.df_transcripts
        
        # 加载形态学图像信息(懒加载，仅获取图像尺寸)
        self.tif_path = os.path.join(data_root, 'morphology.ome.tif')
        with tifffile.TiffFile(self.tif_path) as tif:
            self.img_shape = tif.pages[0].shape
        if init_bar is not None:
            init_bar.update(1)
        try:
            self.tif_memmap = tifffile.memmap(self.tif_path)
        except ValueError:
            self.tif_memmap = None
        if init_bar is not None:
            init_bar.update(1)
        self.tif_array_cache = None
        
        # [关键修复1] 拆分Transform: 避免Compose导致的数据类型错误
        # 单独存储转换函数，在__getitem__中按顺序应用
        self.to_tensor = transforms.ToTensor()  # 转为Tensor并归一化到[0,1]
        self.normalize = transforms.Normalize(  # 标准化操作
            (0.485, 0.456, 0.406),  # ImageNet均值
            (0.229, 0.224, 0.225)   # ImageNet标准差
        )
        
        # 设置数据集长度
        # 训练集生成1000个切片，测试集生成200个切片
        self.length = int(num_samples) if num_samples is not None else (1000 if split == 'train' else 200)

        # 加载细胞边界数据
        print("正在加载边界数据...")
        # 加载细胞边界多边形数据
        self.df_boundaries = pd.read_parquet(os.path.join(data_root, 'cell_boundaries.parquet'))
        grouped = self.df_boundaries.groupby('cell_id')
        boundary_iter = grouped
        if self.show_progress and tqdm is not None:
            boundary_iter = tqdm(grouped, total=grouped.ngroups, desc=f"OVSeg[{self.split}] 边界索引", unit="cell", dynamic_ncols=True)
        self.boundary_dict = {}
        for cell_id, group in boundary_iter:
            self.boundary_dict[cell_id] = group[['vertex_x', 'vertex_y']].to_numpy(np.float32)
        if init_bar is not None:
            init_bar.update(1)
            init_bar.close()
        self.fixed_cell_indices = None
        if self.split in ('val', 'test') and self.deterministic_eval:
            self.fixed_cell_indices = self._build_fixed_cell_indices()

    def __len__(self):
        return self.length

    def _build_fixed_cell_indices(self):
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
            axis = 0 if self.split_mode == 'spatial_x' else 1
            ordered_indices = np.argsort(self.cells_xy[:, axis], kind='mergesort')
            ordered_coord = self.cells_xy[ordered_indices, axis]
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
        if self.fixed_cell_indices is not None:
            cell_idx = int(self.fixed_cell_indices[index])
            return self.cells_xy[cell_idx]
        rand_local_idx = np.random.randint(0, self.split_center_indices.shape[0])
        rand_idx = int(self.split_center_indices[rand_local_idx])
        return self.cells_xy[rand_idx]

    def _select_local_transcripts(self, local_tx_xy, local_tx_gene_ids, local_tx_qv):
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

    def __getitem__(self, index):
        cx_micron, cy_micron = self._get_crop_center(index)
        
        cx_px = int(cx_micron / self.pixel_size)
        cy_px = int(cy_micron / self.pixel_size)
        half_size = self.crop_size // 2
        img_h, img_w = self.img_shape
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
            
        # 记录物理坐标用于后续筛选
        x_min_um, x_max_um = x_min * self.pixel_size, x_max * self.pixel_size
        y_min_um, y_max_um = y_min * self.pixel_size, y_max * self.pixel_size

        # --- 2. 读取图像 Patch ---
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
        
        # [关键修复3] 自动补全尺寸 (Padding)
        # 如果切出来的图小于 crop_size，说明在边缘，强行补黑边
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

        # [关键修复4] 类型转换 (uint16 -> float32)
        img_patch = img_patch.astype('float32') / 65535.0
        
        # [关键修复5] 手动 Transform 流程
        img_tensor = self.to_tensor(img_patch) # [1, 512, 512]
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1) # [3, 512, 512]
        img_tensor = self.normalize(img_tensor)

        # --- 3. 生成 GT Mask ---
        local_mask = (
            (self.cells_xy[:, 0] >= x_min_um) & (self.cells_xy[:, 0] < x_max_um) &
            (self.cells_xy[:, 1] >= y_min_um) & (self.cells_xy[:, 1] < y_max_um)
        ) & self.split_cell_mask
        local_idx = np.nonzero(local_mask)[0]
        valid_cell_ids = self.cell_ids[local_idx]
        local_cells_xy = self.cells_xy[local_idx]

        mask = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)
        instance_masks = []
        instance_target_valid = []

        for cell_id in valid_cell_ids:
            boundary = self.boundary_dict.get(cell_id)
            if boundary is None:
                instance_masks.append(np.zeros((self.crop_size, self.crop_size), dtype=np.uint8))
                instance_target_valid.append(False)
                continue
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

        mask_tensor = torch.from_numpy(mask).long().unsqueeze(0)
        if len(instance_masks) > 0:
            instance_masks_tensor = torch.from_numpy(np.stack(instance_masks, axis=0)).float()
            instance_target_valid_tensor = torch.tensor(instance_target_valid, dtype=torch.bool)
        else:
            instance_masks_tensor = torch.zeros((0, self.crop_size, self.crop_size), dtype=torch.float32)
            instance_target_valid_tensor = torch.zeros((0,), dtype=torch.bool)

        # --- 4. 切分转录本 (Omics) ---
        tx_mask = (
            (self.tx_xy[:, 0] >= x_min_um) & (self.tx_xy[:, 0] < x_max_um) &
            (self.tx_xy[:, 1] >= y_min_um) & (self.tx_xy[:, 1] < y_max_um)
        )
        local_tx_xy = self.tx_xy[tx_mask]
        local_tx_gene_ids = self.tx_gene_ids[tx_mask]
        local_tx_qv = self.tx_qv[tx_mask]
        local_tx_xy, local_tx_gene_ids, local_tx_qv = self._select_local_transcripts(
            local_tx_xy,
            local_tx_gene_ids,
            local_tx_qv,
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
        
        # --- 5. 切分质心 ---
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
            'centroids': centroids,      
            'instance_masks': instance_masks_tensor,
            'instance_target_valid': instance_target_valid_tensor,
            'mask': mask_tensor, 
        }
