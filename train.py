import os
import torch
import torch.nn as nn
import torch.optim as optim
import utils.dataloader as DataSet
import torch.backends.cudnn as cudnn
from nets.vit import vit_base_patch16_384
from utils.utils_fit import fit_one_epoch

# 配置设备，使用1号到6号GPU
# device_ids = [1]  # 指定使用的GPU，不包括0号
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 明确将模型移动到cuda:1
print(f"Using device: {device}")

class train_vit():
    def __init__(self, num_classes, log_file="train_log.txt", checkpoint_path="checkpoint.pth"):
        super(train_vit, self).__init__()

        self.train_txt_path = 'main_txt/train.txt'  # 存放训练集的txt文件路径
        self.test_txt_path = 'main_txt/test.txt'  # 存放测试集的txt文件路径

        # 创建ViT模型
        model = vit_base_patch16_384()  # 创建 ViT 模型
        inchannel = model.head.in_features
        model.head = nn.Linear(inchannel, num_classes)  # 修改输出类别

        # 配置多GPU训练
        if torch.cuda.is_available():
            cudnn.benchmark = True
            model = model.to(device)  # 将模型移到cuda:1
            # model = nn.DataParallel(model, device_ids=device_ids)  # 使用DataParallel并指定device_ids
        self.model = model

        # 定义损失函数
        self.vit_loss = nn.CrossEntropyLoss()
        self.test_accuracy_max = 0  # 初始化最大测试集准确率
        self.log_file = log_file  # 日志文件路径
        self.checkpoint_path = checkpoint_path  # 保存的checkpoint路径

    def save_checkpoint(self, epoch, optimizer, lr_scheduler):
        """保存模型checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'test_accuracy_max': self.test_accuracy_max
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self):
        """加载模型checkpoint"""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.test_accuracy_max = checkpoint['test_accuracy_max']
            print(f"Checkpoint loaded. Resuming training from epoch {checkpoint['epoch']}")
            return checkpoint['epoch']
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0  # 如果没有找到checkpoint，返回epoch为0

    def train(self, batch_size, epochs):
        # 使用分组学习率策略，分类器部分学习率较高，特征提取部分学习率较低
        high_rate_params = []
        low_rate_params = []

        for name, params in self.model.named_parameters():
            if 'head' in name:
                high_rate_params += [params]
            else:
                low_rate_params += [params]

        # 优化器设置
        optimizer = optim.SGD(
            params=[{"params": high_rate_params, 'lr': 0.001},
                    {"params": low_rate_params, 'lr': 0.0002}],  # 使用较低的学习率
            momentum=0.8, weight_decay=5e-4)

        # 学习率下降策略
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3, last_epoch=-1)

        # 加载checkpoint，恢复训练状态
        start_epoch = self.load_checkpoint()

        # 定义训练集与测试集
        train_data = torch.utils.data.DataLoader(DataSet.train_dataset(self.train_txt_path), batch_size=batch_size,
                                                 shuffle=True, num_workers=2)
        test_data = torch.utils.data.DataLoader(DataSet.test_dataset(self.test_txt_path), batch_size=batch_size,
                                                shuffle=False, num_workers=2)

        # 开始训练
        for epoch in range(start_epoch, epochs):
            print(f'\nEpoch: {epoch + 1}/{epochs}')
            # 训练并评估
            train_loss, train_accuracy, test_accuracy = fit_one_epoch(
                self.model, self.vit_loss, optimizer, train_data, test_data, device, epoch + 1, self.log_file)

            # 更新学习率
            lr_scheduler.step()

            # 打印每个epoch的信息
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")

            # 打印当前学习率
            for param_group in optimizer.param_groups:
                print(f"Current Learning Rate: {param_group['lr']}")

            # 保存最优模型
            if test_accuracy > self.test_accuracy_max:
                self.test_accuracy_max = test_accuracy
                self.save_checkpoint(epoch, optimizer, lr_scheduler)  # 保存checkpoint
                print("Checkpoint saved with new best accuracy!")

            # 记录每个epoch的训练与测试结果
            with open(self.log_file, "a") as f:
                f.write(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                        f"Test Accuracy: {test_accuracy:.4f}\n")

if __name__ == "__main__":
    # 设置训练参数
    train = train_vit(10)  # CIFAR-10 数据集的类别数为10
    train.train(1024, 200)  # 训练 10 个 epoch，批次大小为 128
