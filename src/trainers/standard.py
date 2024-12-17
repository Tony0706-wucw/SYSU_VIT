import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from src.utils.utils_fit import fit_one_epoch
from src.utils.logger import Logger
from src.utils.dataloader import Vit_dataset, train_dataset, test_dataset
from tqdm import tqdm
import torch.nn.functional as F

class BaseTrainer:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        self.logger = Logger(config.log_file)
        self.test_accuracy_max = 0

    def _create_dataloaders(self):
        train_loader = torch.utils.data.DataLoader(
            train_dataset(self.config.train_txt_path),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset(self.config.test_txt_path),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader

    def save_checkpoint(self, epoch, optimizer, lr_scheduler=None, save_path=None):
        """保存检查点"""
        if save_path is None:
            save_path = os.path.join(self.config.checkpoint_dir, 'checkpoint.pth')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        # 只有在提供了学习率调度器时才保存其状态
        if lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
        self.logger.info(f'Checkpoint saved to {save_path}')

    def load_checkpoint(self, path):
        """加载检查点"""
        if os.path.exists(path):
            self.logger.info(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.test_accuracy_max = checkpoint['test_accuracy_max']
            self.logger.info(f"Checkpoint loaded. Resuming from epoch {checkpoint['epoch']}")
            return checkpoint
        else:
            self.logger.info("No checkpoint found. Starting from scratch.")
            return None

class StandardTrainer(BaseTrainer):
    """标准训练器：用于基础模型、patch8模型和剪枝模型的训练"""
    
    def __init__(self, config, model, device):
        super().__init__(config, model, device)
        # 将模型移动到指定设备
        self.model = self.model.to(device)
        self.test_accuracy_max = 0.0
        
        # 创建日志目录
        os.makedirs(os.path.dirname(config.log_dir), exist_ok=True)
        
        # 使用模型名称创建日志文件
        if hasattr(config, 'model_name'):
            model_name = config.model_name
        else:
            model_name = 'baseline'
        log_file = os.path.join(config.log_dir, f'train_{model_name}.log')
        self.logger = Logger(log_file)

    def _create_optimizer(self):
        """创建优化器"""
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.high_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        return optimizer

    def train(self):
        """训练模型"""
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = self._create_optimizer()
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma
        )
        
        # 记录模型信息
        self.logger.log_model_info(self.model)
        
        # 获取模型名称
        model_name = self.config.model_name if hasattr(self.config, 'model_name') else 'baseline'
        
        # 尝试加载检查点
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'{model_name}_latest.pth')
        checkpoint = self.load_checkpoint(checkpoint_path)
        start_epoch = checkpoint['epoch'] if checkpoint else 0
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print(f"从检查点恢复训练，开始轮次：{start_epoch}")
        
        train_loader, test_loader = self._create_dataloaders()
        
        for epoch in range(start_epoch, self.config.epochs):
            # 训练一个epoch
            train_loss, train_accuracy, test_loss, test_accuracy = fit_one_epoch(
                self.model, criterion, optimizer,
                train_loader, test_loader,
                self.device, epoch,
                os.path.join(self.config.log_dir, f'train_{model_name}.log')
            )
            
            # 更新学习率
            lr_scheduler.step()
            
            # 保存最佳模型
            if test_accuracy > self.test_accuracy_max:
                self.test_accuracy_max = test_accuracy
                self.save_checkpoint(
                    epoch,
                    optimizer,
                    lr_scheduler,
                    os.path.join(self.config.checkpoint_dir, f'{model_name}_best.pth')
                )
            
            # 定期保存最新模型
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(
                    epoch,
                    optimizer,
                    lr_scheduler,
                    os.path.join(self.config.checkpoint_dir, f'{model_name}_latest.pth')
                )

class TeacherTrainer(BaseTrainer):
    """教师模型训练器：用于训练作为知识蒸馏源的教师模型"""
    
    def train(self, train_loader, epochs=10):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.high_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        for epoch in range(epochs):
            running_loss = 0.0
            batch_count = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batch_count += 1
            
            avg_loss = running_loss / batch_count if batch_count > 0 else float('inf')
            self.logger.info(f'Teacher Training - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

class DistillationTrainer(BaseTrainer):
    """知识蒸馏训练器：用于通过知识蒸馏训练学生模型"""
    
    def __init__(self, config, student_model, teacher_model, device):
        super().__init__(config, student_model, device)
        self.teacher_model = teacher_model
        self.teacher_model.eval()  # 教师模型设置为评估模式
        self.temperature = config.temperature
        self.alpha = config.alpha if hasattr(config, 'alpha') else 0.5
        
        # 设置logger
        log_file = os.path.join('outputs', 'logs', f'train_distill.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = Logger(log_file)

        # 创建优化器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.high_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )

    def distillation_loss(self, student_outputs, teacher_outputs, labels):
        T = self.temperature
        alpha = self.alpha
        
        # 软目标损失（KL散度）
        soft_targets = F.softmax(teacher_outputs / T, dim=1)
        student_log_softmax = F.log_softmax(student_outputs / T, dim=1)
        distillation_loss = F.kl_div(student_log_softmax, soft_targets, reduction='batchmean') * (T * T)
        
        # 硬目标损失（交叉熵）
        student_loss = F.cross_entropy(student_outputs, labels)
        
        # 组合损失
        total_loss = alpha * student_loss + (1 - alpha) * distillation_loss
        return total_loss

    def evaluate(self):
        """评估模型在测试集上的性能"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        self.model.train()
        return acc

    def train(self, epochs=100):
        train_loader, test_loader = self._create_dataloaders()
        self.test_loader = test_loader
        
        best_acc = 0.0
        self.model.train()
        self.teacher_model.eval()

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            # 使用tqdm创建进度条
            pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]')
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 获取教师模型和学生模型的输出
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                student_outputs = self.model(images)
                
                # 计算蒸馏损失
                loss = self.distillation_loss(student_outputs, teacher_outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = student_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
            
            # 计算epoch的平均损失和准确率
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            
            # 记录训练信息
            self.logger.info(f'Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
            
            # 每个epoch进行验证
            test_acc = self.evaluate()
            
            # 保存最佳模型和最新模型
            if test_acc > best_acc:
                best_acc = test_acc
                self.save_checkpoint(
                    epoch,
                    self.optimizer,
                    None,
                    os.path.join(self.config.checkpoint_dir, f'distill_best.pth')
                )
            self.save_checkpoint(
                epoch,
                self.optimizer,
                None,
                os.path.join(self.config.checkpoint_dir, f'distill_latest.pth')
            )

            self.logger.info(f'Test Acc: {test_acc:.2f}%, Best Acc: {best_acc:.2f}%')
