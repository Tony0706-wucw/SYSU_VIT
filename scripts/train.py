import os
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from nets.vit import vit_patch8_emb256_dep2
from nets.vit import vit_baseline
from nets.vit_pruned import vit_pruned
from trainers import StandardTrainer, TeacherTrainer, DistillationTrainer

class VitTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()

    def _create_model(self):
        if self.config.model_type == 'baseline':
            model = vit_baseline()  # 使用baseline模型
        elif self.config.model_type == 'patch8':
            model = vit_patch8_emb256_dep2()  # 使用patch8的模型
        elif self.config.model_type == 'pruned':
            model = vit_pruned()  # 使用剪枝模型
        else:
            raise ValueError("Invalid model type selected. Choose 'baseline', 'patch8', or 'pruned'.")

        inchannel = model.head.in_features
        model.head = nn.Linear(inchannel, self.config.num_classes)
        
        if torch.cuda.is_available():
            cudnn.benchmark = True
            model = model.to(self.device)
        return model

    def train_standard(self):
        """标准训练方法"""
        trainer = StandardTrainer(self.config, self.model, self.device)
        trainer.train()

    def train_as_teacher(self, train_loader, epochs=10):
        """作为教师模型进行训练"""
        trainer = TeacherTrainer(self.config, self.model, self.device)
        trainer.train(train_loader, epochs)

    def train_with_distillation(self, teacher_model, epochs=10):
        """使用知识蒸馏进行训练"""
        trainer = DistillationTrainer(self.config, self.model, teacher_model, self.device)
        trainer.train(epochs)

if __name__ == "__main__":
    from config import config
    
    trainer = VitTrainer(config)
    
    # 根据需要选择不同的训练方式
    if config.model_type == 'pruned':
        # 首先训练教师模型
        teacher_trainer = VitTrainer(config)
        teacher_trainer.config.model_type = 'baseline'
        teacher_model = teacher_trainer._create_model()
        teacher_trainer.train_standard()  # 先训练教师模型
        
        # 然后使用知识蒸馏训练剪枝模型
        trainer.train_with_distillation(teacher_model)
    else:
        # 使用标准训练方式
        trainer.train_standard()
