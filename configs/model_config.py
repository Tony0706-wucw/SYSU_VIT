from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """模型配置类"""
    # 模型类型
    model_type: str = 'baseline'  # 'baseline', 'patch8', 'pruned'
    
    # 模型架构配置
    img_size: int = 32
    patch_size: int = 8
    in_chans: int = 3
    embed_dim: int = 256
    depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    num_classes: int = 10
    
    # 剪枝配置
    pruning_amount: float = 0.4
    pruning_checkpoint_path: str = "outputs/checkpoints/checkpoint_pruned.pth"
