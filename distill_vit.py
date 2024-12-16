import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

from nets.vit import VisionTransformer
from train import VitTrainer
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载教师模型的权重
teacher_model = VisionTransformer(embed_dim=512, depth=12, num_heads=8).to(device)
teacher_model.load_state_dict(torch.load("teacher_model.pth", map_location=device))

# 创建VitTrainer实例
trainer = VitTrainer(config)

# 训练教师模型
trainer.model = VisionTransformer(embed_dim=512, depth=12, num_heads=8).to(device)
print("\n开始训练教师模型...")
trainer.train_teacher(trainer.model, train_loader, trainer._create_optimizer(), epochs=10)
torch.save(trainer.model.state_dict(), "teacher_model.pth")
print("教师模型已保存为 'teacher_model.pth'")

# 训练学生模型
trainer.model = VisionTransformer(embed_dim=256, depth=2, num_heads=4).to(device)
print("\n开始训练学生模型...")
trainer.train_distillation(trainer.model, teacher_model, train_loader, trainer._create_optimizer(), alpha=0.7, temperature=3.0, epochs=10)
torch.save(trainer.model.state_dict(), "student_model.pth")
print("学生模型已保存为 'student_model.pth'")
