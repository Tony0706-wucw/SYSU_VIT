import os
import sys
import argparse

# 添加项目根目录到Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from configs import TrainConfig
from src.trainers.standard import StandardTrainer
from src.models.vit import vit_baseline, vit_patch8_emb256_dep2
from src.models.vit_pruned import vit_pruned

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
    config.model.model_type = args.model
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    
    # 根据模式运行
    if args.mode == 'train':
        trainer = StandardTrainer(config)
        trainer.train()
    
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            raise ValueError("评估模式需要提供检查点路径")
        trainer = StandardTrainer(config)
        trainer.evaluate()
    
    elif args.mode == 'distill':
        # 首先训练教师模型（如果没有提供检查点）
        if not args.checkpoint:
            print("训练教师模型...")
            teacher_config = TrainConfig()
            teacher_config.model.model_type = 'baseline'
            teacher_trainer = StandardTrainer(teacher_config)
            teacher_trainer.train()
            teacher_model = teacher_trainer.model
        else:
            print(f"从检查点加载教师模型: {args.checkpoint}")
            teacher_model = vit_baseline()
            teacher_model.load_state_dict(torch.load(args.checkpoint))
        
        # 然后进行知识蒸馏
        print("开始知识蒸馏...")
        config.model.model_type = 'pruned'  # 使用剪枝模型作为学生模型
        trainer = StandardTrainer(config)
        trainer.train_with_distillation(teacher_model)

if __name__ == '__main__':
    main()
