import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


###### 定义数据读取 ######
# 将输入图像转化为RGB形式
def default_loader(path):
    # 规范化路径，确保跨平台兼容
    # 替换反斜杠为正斜杠，适应 Linux 路径

    normalized_path = os.path.normpath(path)
    normalized_path = path.replace("\\", "/")
    # 将路径打印出来供调试
    # print(f"Attempting to load image from: {normalized_path}")

    # 检查文件是否存在
    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"File not found: {normalized_path}")

    # 处理异常，防止文件损坏或无法读取时崩溃
    try:
        img = Image.open(normalized_path).convert('RGB')
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {normalized_path} with error: {e}")

# 定义数据读取类
class Vit_dataset(Dataset):
    def __init__(self, txt, transform=None, loader=default_loader):
        super(Vit_dataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除本行末尾的分行字符
            words = line.split()  # 分割图片路径及标签
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # 规范化路径，适配不同操作系统
        normalized_path = os.path.normpath(fn)
        img = self.loader(normalized_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


# # 定义预处理
# data_transform = {
#     "train": transforms.Compose([
#         transforms.Resize(438),
#         # transforms.Resize(224),
#         transforms.RandomResizedCrop(384),
#         # transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ]),
#     "test": transforms.Compose([
#         transforms.Resize(438),
#         # transforms.Resize(224),
#         transforms.CenterCrop(384),
#         # transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
#
data_transform = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
    ]),
    "test": transforms.Compose([
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
    ])
}

def train_dataset(root):
    return Vit_dataset(root, transform=data_transform["train"])


def test_dataset(root):
    return Vit_dataset(root, transform=data_transform["test"])
