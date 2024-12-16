import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from nets.vit_pruned import vit_pruned

def main():
    # 创建模型
    model = vit_pruned()
    print("Model Architecture:")
    print(model)
    
    # 创建随机输入进行测试
    batch_size = 1
    x = torch.randn(batch_size, 3, 32, 32)
    
    # 测试前向传播
    output = model(x)
    print(f"\nForward Pass Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 分析剪枝效果
    print("\nPruning Analysis:")
    total_zeros = 0
    total_params = 0
    
    for i, block in enumerate(model.blocks):
        print(f"\nBlock {i+1}:")
        # 分析fc1的权重
        if hasattr(block.mlp.fc1, 'weight_mask'):
            zeros = (block.mlp.fc1.weight_mask == 0).sum().item()
            total = block.mlp.fc1.weight_mask.numel()
            sparsity = zeros/total
            print(f"  MLP FC1: {block.mlp.fc1.weight.shape}")
            print(f"  - Parameters: {total:,}")
            print(f"  - Zeros: {zeros:,}")
            print(f"  - Sparsity: {sparsity:.2%}")
            total_zeros += zeros
            total_params += total
            
        # 分析fc2的权重
        if hasattr(block.mlp.fc2, 'weight_mask'):
            zeros = (block.mlp.fc2.weight_mask == 0).sum().item()
            total = block.mlp.fc2.weight_mask.numel()
            sparsity = zeros/total
            print(f"  MLP FC2: {block.mlp.fc2.weight.shape}")
            print(f"  - Parameters: {total:,}")
            print(f"  - Zeros: {zeros:,}")
            print(f"  - Sparsity: {sparsity:.2%}")
            total_zeros += zeros
            total_params += total
    
    if total_params > 0:
        print(f"\nOverall MLP Layers Analysis:")
        print(f"Total parameters: {total_params:,}")
        print(f"Total zeros: {total_zeros:,}")
        print(f"Overall sparsity: {total_zeros/total_params:.2%}")
        print(f"Parameter reduction: {total_zeros:,} parameters")

if __name__ == '__main__':
    main()
    