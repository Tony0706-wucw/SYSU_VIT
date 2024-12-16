# Vision Transformer Training Framework

这是一个用于训练Vision Transformer的框架，支持标准训练、知识蒸馏和模型剪枝。

## 功能特点

- 支持多种ViT模型变体（baseline、patch8、pruned）
- 实现知识蒸馏训练
- 支持模型剪枝
- 完善的日志系统
- 灵活的配置管理

## 项目结构

```
Vit/
├── configs/          # 配置文件
│   ├── base_config.py   # 基础配置
│   ├── model_config.py  # 模型配置
│   └── train_config.py  # 训练配置
├── src/             # 源代码
│   ├── models/      # 模型定义
│   ├── trainers/    # 训练器
│   └── utils/       # 工具函数
├── scripts/         # 运行脚本
├── outputs/         # 输出目录
│   ├── checkpoints/ # 模型检查点
│   └── logs/       # 训练日志
└── setup.py        # 安装脚本
```

## 安装

1. 克隆仓库：
```bash
git clone [repository-url]
cd Vit
```

2. 安装依赖：
```bash
pip install -e .
```

## 使用方法

### 1. 标准训练

```bash
# 训练基础模型
python scripts/run.py --mode train --model baseline

# 训练patch8模型
python scripts/run.py --mode train --model patch8 --batch-size 512

# 训练剪枝模型
python scripts/run.py --mode train --model pruned --epochs 200
```

### 2. 知识蒸馏

```bash
# 从头开始知识蒸馏（先训练教师模型，再进行蒸馏）
python scripts/run.py --mode distill

# 使用已有的教师模型进行蒸馏
python scripts/run.py --mode distill --checkpoint outputs/checkpoints/checkpoint_baseline.pth
```

### 3. 评估模型

```bash
# 评估训练好的模型
python scripts/run.py --mode evaluate --model baseline --checkpoint outputs/checkpoints/checkpoint_baseline.pth
```

## 配置说明

### 1. 基础配置 (base_config.py)
- 路径配置
- 设备配置
- 随机种子设置

### 2. 模型配置 (model_config.py)
- 模型类型选择
- 模型架构参数
- 剪枝配置

### 3. 训练配置 (train_config.py)
- 训练超参数
- 优化器设置
- 知识蒸馏参数

可以通过修改配置文件或使用命令行参数来自定义这些设置。

## 日志系统

训练过程中会记录详细的信息：

1. **训练过程信息**：
   - 时间戳
   - 训练轮次（Epoch）
   - 训练损失（Training Loss）
   - 验证损失（Validation Loss）
   - 准确率（Accuracy）
   - 学习率（Learning Rate）
   - 梯度信息（Gradient Information）

2. **日志位置**：
   - 训练日志：`outputs/logs/train_[model_type].log`
   - 蒸馏日志：`outputs/logs/distill_[model_type].log`

3. **检查点保存**：
   - 位置：`outputs/checkpoints/`
   - 保存策略：保存最佳模型和定期保存

## 开发指南

### 添加新模型

1. 在 `src/models/` 下创建新的模型文件
2. 在 `configs/model_config.py` 中添加相应配置
3. 更新 `scripts/run.py` 中的模型选择逻辑

### 添加新的训练策略

1. 在 `src/trainers/` 下创建新的训练器类
2. 在 `configs/train_config.py` 中添加相应配置
3. 更新 `scripts/run.py` 中的训练模式

## 调试建议

1. **快速测试**：
   - 使用小的 batch_size
   - 减少训练轮数
   - 使用较小的数据集

2. **断点续训**：
   - 使用 `--checkpoint` 参数加载之前的模型
   - 检查日志文件追踪训练过程

3. **内存优化**：
   - 适当调整 batch_size
   - 使用梯度累积
   - 启用混合精度训练

## 常见问题

1. **内存不足**：
   - 减小 batch_size
   - 检查数据预处理是否释放内存
   - 使用梯度累积

2. **训练不收敛**：
   - 检查学习率设置
   - 查看梯度是否正常
   - 尝试调整模型架构

3. **GPU利用率低**：
   - 增加 batch_size
   - 检查数据加载瓶颈
   - 优化数据预处理流程

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 许可证

[MIT License](LICENSE)
