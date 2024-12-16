import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from utils.utils_fit import fit_one_epoch
from logger import Logger

class BaseTrainer:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        self.logger = Logger(config.log_file)
        self.test_accuracy_max = 0

    def _create_dataloaders(self):
        train_data = torch.utils.data.DataLoader(
            DataSet.train_dataset(self.config.train_txt_path),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        test_data = torch.utils.data.DataLoader(
            DataSet.test_dataset(self.config.test_txt_path),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        return train_data, test_data

    def save_checkpoint(self, epoch, optimizer, lr_scheduler, path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'test_accuracy_max': self.test_accuracy_max
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, path):
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
    
    def _create_optimizer(self):
        # 使用分组学习率策略
        high_rate_params = []
        low_rate_params = []
        
        for name, params in self.model.named_parameters():
            if 'head' in name:
                high_rate_params += [params]
            else:
                low_rate_params += [params]
                
        optimizer = optim.SGD(
            params=[
                {"params": high_rate_params, 'lr': self.config.high_lr},
                {"params": low_rate_params, 'lr': self.config.low_lr}
            ],
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        return optimizer

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = self._create_optimizer()
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma
        )
        
        # 记录模型信息
        self.logger.log_model_info(self.model)
        
        checkpoint = self.load_checkpoint(self.config.checkpoint_path)
        start_epoch = checkpoint['epoch'] if checkpoint else 0
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
        train_loader, test_loader = self._create_dataloaders()
        total_steps = len(train_loader)
        
        for epoch in range(start_epoch, self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for step, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # 计算梯度范数
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()
                epoch_loss += loss.item()
                
                # 获取当前学习率
                lr_dict = {
                    'head': optimizer.param_groups[0]['lr'],
                    'base': optimizer.param_groups[1]['lr']
                }
                
                # 记录每个步骤的信息
                if step % 10 == 0:  # 每10步记录一次
                    step_metrics = {
                        'Loss': loss.item(),
                        'Accuracy': 100 * epoch_correct / epoch_total
                    }
                    self.logger.log_training_step(
                        epoch + 1, step + 1, total_steps,
                        step_metrics, lr_dict, grad_norm
                    )
            
            # 计算训练集上的平均指标
            avg_train_loss = epoch_loss / len(train_loader)
            train_accuracy = 100 * epoch_correct / epoch_total
            
            # 在测试集上评估
            self.model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            avg_test_loss = test_loss / len(test_loader)
            test_accuracy = 100 * test_correct / test_total
            
            # 记录每个epoch的总结信息
            epoch_metrics = {
                'Loss': avg_train_loss,
                'Accuracy': train_accuracy,
                'Learning Rate (head)': lr_dict['head'],
                'Learning Rate (base)': lr_dict['base']
            }
            
            validation_metrics = {
                'Loss': avg_test_loss,
                'Accuracy': test_accuracy
            }
            
            self.logger.log_epoch_summary(epoch + 1, epoch_metrics, validation_metrics)
            
            # 更新学习率
            lr_scheduler.step()
            
            # 保存最佳模型
            if test_accuracy > self.test_accuracy_max:
                self.test_accuracy_max = test_accuracy
                self.save_checkpoint(epoch, optimizer, lr_scheduler, self.config.checkpoint_path)

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

    def distillation_loss(self, student_outputs, teacher_outputs, labels):
        T = self.config.temperature
        alpha = self.config.distillation_alpha
        
        # 软目标损失（KL散度）
        soft_targets = nn.functional.softmax(teacher_outputs / T, dim=1)
        student_log_softmax = nn.functional.log_softmax(student_outputs / T, dim=1)
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(student_log_softmax, soft_targets) * (T * T)
        
        # 硬目标损失（交叉熵）
        student_loss = nn.CrossEntropyLoss()(student_outputs, labels)
        
        # 组合损失
        total_loss = alpha * student_loss + (1 - alpha) * distillation_loss
        return total_loss

    def train(self, epochs=10):
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.high_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        train_loader, _ = self._create_dataloaders()
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                # 获取教师模型的输出
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                
                # 获取学生模型的输出并计算损失
                student_outputs = self.model(images)
                loss = self.distillation_loss(student_outputs, teacher_outputs, labels)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            self.logger.info(f'Distillation Training - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            
            # 保存检查点
            if (epoch + 1) % 5 == 0:  # 每5个epoch保存一次
                self.save_checkpoint(epoch, optimizer, None, self.config.distillation_checkpoint_path)
