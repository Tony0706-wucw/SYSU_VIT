import os
import sys
import argparse
import torch

# 添加项目根目录到Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from configs.train_config import TrainConfig
from configs.model_config import ModelConfig
from src.trainers.standard import StandardTrainer, DistillationTrainer
from src.models.nets.vit import vit_baseline, vit_patch8_emb256_dep2
from src.models.nets.vit_pruned import vit_pruned

def parse_args():
    parser = argparse.ArgumentParser(description='Vision Transformer Training')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'distill'],
                        help='运行模式：训练、评估或知识蒸馏')
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'patch8', 'pruned'],
                        help='模型类型')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点路径（用于继续训练或评估）')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建配置
    config = TrainConfig()
    
    # 更新配置
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
        
    # 设置模型配置
    model_config = ModelConfig()
    model_config.model_type = args.model
    
    # 设置模型名称用于日志文件
    config.model_name = args.model
    
    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    if args.mode == 'train':
        # 创建模型
        if args.model == 'baseline':
            model = vit_baseline()
        elif args.model == 'patch8':
            model = vit_patch8_emb256_dep2()
        elif args.model == 'pruned':
            model = vit_pruned()
        else:
            raise ValueError(f"未知的模型类型: {args.model}")
        
        # 创建训练器
        trainer = StandardTrainer(config, model, device)
        trainer.train()
        
    elif args.mode == 'distill':
        # 创建教师模型（使用基础模型）
        teacher_model = vit_baseline()
        teacher_checkpoint = os.path.join(config.checkpoint_dir, 'baseline_best.pth')
        if not os.path.exists(teacher_checkpoint):
            raise ValueError("找不到教师模型的检查点，请先训练基础模型")
        
        # 加载教师模型的权重
        checkpoint = torch.load(teacher_checkpoint)
        teacher_model.load_state_dict(checkpoint['model_state_dict'])
        teacher_model = teacher_model.to(device)
        
        # 创建学生模型（使用剪枝模型）
        student_model = vit_pruned()
        student_model = student_model.to(device)  # 将学生模型移动到设备上
        
        # 设置蒸馏相关的配置
        config.temperature = 3.0  # 温度参数
        config.distillation_alpha = 0.5  # 蒸馏损失权重
        config.distillation_checkpoint_path = os.path.join(config.checkpoint_dir, 'distilled_model.pth')
        
        # 创建蒸馏训练器
        trainer = DistillationTrainer(config, student_model, teacher_model, device)
        trainer.train(epochs=args.epochs or 100)
        
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError("评估模式需要提供检查点路径")
            
        if args.model == 'baseline':
            model = vit_baseline()
        elif args.model == 'patch8':
            model = vit_patch8_emb256_dep2()
        elif args.model == 'pruned':
            model = vit_pruned()
        else:
            raise ValueError(f"不支持的模型类型: {args.model}")
            
        model = model.to(device)
        
        # 加载检查点
        print(f"加载检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建训练器并评估
        trainer = StandardTrainer(config, model, device)
        test_loader = trainer._create_dataloaders()[1]
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        print(f"模型在测试集上的准确率: {accuracy:.2f}%")
        
if __name__ == '__main__':
    main()
