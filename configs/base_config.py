from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseConfig:
    """基础配置类"""
    # 路径配置
    data_dir: str = 'data'
    output_dir: str = 'outputs'
    checkpoint_dir: str = 'outputs/checkpoints'
    log_dir: str = 'outputs/logs'
    
    # 设备配置
    device: str = 'cuda'  # 'cuda' or 'cpu'
    num_workers: int = 2
    
    # 随机种子
    seed: int = 42
    
    def __post_init__(self):
        """初始化后的处理"""
        import os
        # 确保必要的目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
