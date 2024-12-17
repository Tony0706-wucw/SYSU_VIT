import os
import time
import torch
import numpy as np
from typing import Dict, Any

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.metrics_history = {}
        
        # Create log file if it doesn't exist
        if not os.path.exists(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'w') as f:
                f.write(f'Log started at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    def log(self, message, level="INFO"):
        """Write a message to the log file and print it to console"""
        with open(self.log_file, 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            log_message = f'[{timestamp}] [{level}] {message}\n'
            f.write(log_message)
            print(log_message, end='')

    def info(self, message):
        """Log an info message"""
        self.log(message, "INFO")
    
    def warning(self, message):
        """Log a warning message"""
        self.log(message, "WARNING")
    
    def error(self, message):
        """Log an error message"""
        self.log(message, "ERROR")
    
    def __call__(self, message):
        """Allow using the logger instance directly as a function"""
        self.info(message)

    def log_training_step(self, epoch: int, step: int, total_steps: int, metrics: Dict[str, Any],
                         learning_rates: Dict[str, float], grad_norm: float = None):
        """
        记录训练步骤的详细信息
        Args:
            epoch: 当前训练轮次
            step: 当前步骤
            total_steps: 总步骤数
            metrics: 包含损失值、准确率等指标的字典
            learning_rates: 不同参数组的学习率字典
            grad_norm: 梯度范数（可选）
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 构建基本信息
        log_parts = [
            f'[{timestamp}]',
            f'[TRAIN]',
            f'Epoch: {epoch}',
            f'Step: [{step}/{total_steps}]'
        ]
        
        # 添加指标信息
        for name, value in metrics.items():
            if isinstance(value, (float, np.float32, np.float64)):
                log_parts.append(f'{name}: {value:.4f}')
            else:
                log_parts.append(f'{name}: {value}')
        
        # 添加学习率信息
        for param_group, lr in learning_rates.items():
            log_parts.append(f'LR_{param_group}: {lr:.6f}')
        
        # 添加梯度信息
        if grad_norm is not None:
            log_parts.append(f'Grad Norm: {grad_norm:.4f}')
        
        # 组合并记录信息
        log_message = ' | '.join(log_parts)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
        print(log_message)
        
        # 更新指标历史
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)

    def log_epoch_summary(self, epoch: int, epoch_metrics: Dict[str, Any], 
                         validation_metrics: Dict[str, Any] = None):
        """
        记录每个训练轮次的总结信息
        Args:
            epoch: 当前训练轮次
            epoch_metrics: 训练集上的指标
            validation_metrics: 验证集上的指标（可选）
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 构建训练指标信息
        log_parts = [
            f'[{timestamp}]',
            f'[EPOCH SUMMARY]',
            f'Epoch: {epoch}'
        ]
        
        # 添加训练指标
        log_parts.append('Training Metrics:')
        for name, value in epoch_metrics.items():
            if isinstance(value, (float, np.float32, np.float64)):
                log_parts.append(f'  {name}: {value:.4f}')
            else:
                log_parts.append(f'  {name}: {value}')
        
        # 添加验证指标（如果有）
        if validation_metrics:
            log_parts.append('Validation Metrics:')
            for name, value in validation_metrics.items():
                if isinstance(value, (float, np.float32, np.float64)):
                    log_parts.append(f'  {name}: {value:.4f}')
                else:
                    log_parts.append(f'  {name}: {value}')
        
        # 组合并记录信息
        log_message = '\n'.join(log_parts)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n\n')  # 添加额外的换行使输出更清晰
        print(log_message + '\n')

    def log_model_info(self, model: torch.nn.Module):
        """记录模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = [
            "[MODEL INFO]",
            f"Architecture: {model.__class__.__name__}",
            f"Total Parameters: {total_params:,}",
            f"Trainable Parameters: {trainable_params:,}",
        ]
        
        with open(self.log_file, "a") as f:
            f.write("\n".join(info) + "\n\n")
            
        # 只在控制台打印简要信息
        print(f"Model: {model.__class__.__name__} (Params: {total_params:,})")

    def log_metrics(self, metrics: Dict[str, Any]):
        """记录指标，保持向后兼容"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            for key, value in metrics.items():
                log_message = f'[{timestamp}] [METRIC] {key}: {value}\n'
                f.write(log_message)
                print(log_message, end='')
