import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import Logger
from .base_trainer import VitTrainer

class DistillationTrainer(VitTrainer):
    def __init__(self, config, teacher_model):
        super().__init__(config)
        self.teacher_model = teacher_model
        self.teacher_model.eval()  # 设置教师模型为评估模式
        if torch.cuda.is_available():
            self.teacher_model = self.teacher_model.to(self.device)
        
        # 使用蒸馏专用的日志文件
        self.logger = Logger(config.distillation_log_file)
        
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """计算知识蒸馏损失"""
        # 硬目标损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 软目标损失
        T = self.config.temperature
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)
        
        # 总损失
        total_loss = (1 - self.config.distillation_alpha) * hard_loss + \
                    self.config.distillation_alpha * soft_loss
        
        return total_loss, hard_loss, soft_loss
    
    def train_one_epoch(self, epoch, train_loader, test_loader, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 获取教师模型和学生模型的输出
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)
            student_outputs = self.model(inputs)
            
            # 计算损失
            loss, hard_loss, soft_loss = self.distillation_loss(
                student_outputs, teacher_outputs, targets
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()
            
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                self.logger.info(f'Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} '
                               f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                               f'Loss: {loss.item():.6f}')
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        avg_hard_loss = total_hard_loss / len(train_loader)
        avg_soft_loss = total_soft_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, avg_hard_loss, avg_soft_loss, accuracy
    
    def train(self):
        """知识蒸馏训练主循环"""
        optimizer = self._create_optimizer()
        lr_scheduler = self._create_scheduler(optimizer)
        
        # 加载checkpoint
        checkpoint = self.load_checkpoint()
        start_epoch = checkpoint['epoch'] if checkpoint else 0
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        train_loader, test_loader = self._create_dataloaders()
        
        for epoch in range(start_epoch, self.config.epochs):
            # 训练一个epoch
            train_loss, hard_loss, soft_loss, train_acc = self.train_one_epoch(
                epoch, train_loader, test_loader, optimizer
            )
            
            # 评估
            test_loss, test_acc = self.evaluate(test_loader)
            
            # 记录指标
            self.logger.log_metrics({
                'train/loss': train_loss,
                'train/hard_loss': hard_loss,
                'train/soft_loss': soft_loss,
                'train/accuracy': train_acc,
                'test/loss': test_loss,
                'test/accuracy': test_acc
            }, epoch)
            
            # 更新学习率
            lr_scheduler.step()
            
            # 保存最佳模型
            if test_acc > self.test_accuracy_max:
                self.test_accuracy_max = test_acc
                self.save_checkpoint(epoch, optimizer, lr_scheduler)
                
            self.logger.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                           f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
