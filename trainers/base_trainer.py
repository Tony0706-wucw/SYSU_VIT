import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.logger import Logger
import utils.dataloader as DataSet
from nets.vit import vit_baseline

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = Logger(config.log_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_accuracy_max = 0
        
    def _create_dataloaders(self):
        """创建数据加载器"""
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
    
    def _create_optimizer(self):
        """创建优化器"""
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
    
    def _create_scheduler(self, optimizer):
        """创建学习率调度器"""
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_gamma
        )
    
    def save_checkpoint(self, epoch, optimizer, lr_scheduler):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'test_accuracy_max': self.test_accuracy_max
        }
        torch.save(checkpoint, self.config.checkpoint_path)
        self.logger.info(f"Checkpoint saved at epoch {epoch}")
        
    def load_checkpoint(self):
        """加载检查点"""
        if os.path.exists(self.config.checkpoint_path):
            checkpoint = torch.load(self.config.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.test_accuracy_max = checkpoint['test_accuracy_max']
            self.logger.info(f"Checkpoint loaded. Resuming from epoch {checkpoint['epoch']}")
            return checkpoint
        else:
            self.logger.info("No checkpoint found. Starting from scratch.")
            return None
            
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        return test_loss, accuracy
