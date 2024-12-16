from dataclasses import dataclass
from typing import Optional
from .base_config import BaseConfig
from .model_config import ModelConfig

@dataclass
class TrainConfig(BaseConfig):
    """训练配置类"""
    # 继承基础配置
    
    # 数据集配置
    train_txt_path: str = 'data/train.txt'
    test_txt_path: str = 'data/test.txt'
    batch_size: int = 1024
    
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
    
    # 检查点配置
    save_freq: int = 5  # 每多少个epoch保存一次
    
    # 模型配置
    model: ModelConfig = ModelConfig()
    
    def __post_init__(self):
        """初始化后的处理"""
        super().__post_init__()
        
        # 设置文件路径
        self.log_file = f"{self.log_dir}/train_{self.model.model_type}.log"
        self.checkpoint_path = f"{self.checkpoint_dir}/checkpoint_{self.model.model_type}.pth"
        self.distillation_log_file = f"{self.log_dir}/distill_{self.model.model_type}.log"
        self.distillation_checkpoint_path = f"{self.checkpoint_dir}/distill_{self.model.model_type}.pth"
