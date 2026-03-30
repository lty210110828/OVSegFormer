from .v2_dataset import V2Dataset, get_v2_pallete
from .s4_dataset import S4Dataset
from .ms3_dataset import MS3Dataset
# 1. 确保导入了您的类
from .ovseg_dataset import OVSegDataset 
from mmcv import Config


def build_dataset(type, split, **kwargs):
    if type == 'V2Dataset':
        return V2Dataset(split=split, cfg=Config(kwargs))
    elif type == 'S4Dataset':
        return S4Dataset(split=split, cfg=Config(kwargs))
    elif type == 'MS3Dataset':
        return MS3Dataset(split=split, cfg=Config(kwargs))
    
    # 2. [新增] 必须在这里加上 OVSegDataset 的分支
    elif type == 'OVSegDataset':
        dataset_kwargs = dict(kwargs)
        dataset_kwargs.pop('batch_size', None)
        return OVSegDataset(split=split, **dataset_kwargs)
        
    else:
        raise ValueError(f"Unknown dataset type: {type}") # 建议把报错信息写清楚一点，方便调试


__all__ = ['build_dataset', 'get_v2_pallete']
