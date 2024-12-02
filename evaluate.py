import torch
import torch.nn as nn
import torch.utils.data
from nets.vit import vit_base_patch16_384
import utils.dataloader as DataSet  # 确保路径和模块名称正确

# 设置设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 加载模型
model = vit_base_patch16_384()
model.head = nn.Linear(model.head.in_features, 10)  # 修改为与训练时相同的类别数量
model.load_state_dict(torch.load('vit_base_patch16_384.pth'))  # 确保使用训练时保存的模型路径
model.to(device)  # 确保模型在正确的设备上
model.eval()
# 在测试数据集上进行推理并计算准确率
test_data = torch.utils.data.DataLoader(DataSet.test_dataset('main_txt/test.txt'), batch_size=16, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_data:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        print("predict:"+predicted+"labels:"+labels)
        correct += (predicted == labels).sum().item()

# 打印测试准确率
accuracy = 100 * correct / total
print(f"Test Accuracy on loaded model: {accuracy:.2f}%")
