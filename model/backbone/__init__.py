from .resnet import B2_ResNet
from .pvt import pvt_v2_b5


def build_backbone(type, **kwargs):
    if type == 'res50' or type == 'resnet50':
        return B2_ResNet(**kwargs)
    elif type=='pvt_v2_b5':
        return pvt_v2_b5(**kwargs)
    else:
        # 报错提示，防止未来再出现 NoneType 错误
        raise ValueError(f"Error: Unknown backbone type '{type}'. Please check your config file.")
    
__all__=['build_backbone']
