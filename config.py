from dataclasses import dataclass
from typing import Tuple, Optional
import os

@dataclass
class TrainingConfig:
    # 数据集配置
    train_txt_path: str = 'main_txt/train.txt'
    test_txt_path: str = 'main_txt/test.txt'
    num_classes: int = 10
    batch_size: int = 1024
    num_workers: int = 2

    # 模型配置
    model_type: str = 'baseline'  # 可选择 'baseline'、'patch8' 或 'pruned'
    img_size: int = 32
    patch_size: int = 8
    in_chans: int = 3
    embed_dim: int = 256
    depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    
    # 剪枝配置
    pruning_amount: float = 0.4  # 剪枝比例
    pruning_checkpoint_path: str = "checkpoints/checkpoint_pruned.pth"
    
    # 训练配置
    epochs: int = 300
    high_lr: float = 0.001
    low_lr: float = 0.0002
    momentum: float = 0.8
    weight_decay: float = 5e-4
    lr_step_size: int = 30
    lr_gamma: float = 0.3
    
    # 知识蒸馏配置
    distillation_alpha: float = 0.5
    temperature: float = 3.0
    
    # 文件路径配置
    log_file: str = ""
    checkpoint_path: str = ""
    distillation_log_file: str = ""
    distillation_checkpoint_path: str = ""

    def __post_init__(self):
        # 根据模型类型设置对应的日志和检查点文件名
        model_name = self.model_type
        self.log_file = f"logs/train_log_{model_name}.txt"
        self.checkpoint_path = f"checkpoints/checkpoint_{model_name}.pth"
        
        # 为蒸馏设置特定的文件名
        self.distillation_log_file = f"logs/train_log_{model_name}_distillation.txt"
        self.distillation_checkpoint_path = f"checkpoints/checkpoint_{model_name}_distillation.pth"
        
        # 确保目录存在
        os.makedirs("logs", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

config = TrainingConfig()
