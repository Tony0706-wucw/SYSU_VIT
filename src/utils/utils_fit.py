import time
import torch
import numpy as np
from tqdm import tqdm

def fit_one_epoch(model, vit_loss, optimizer, train_data, test_data, device, epoch, log_file):
    model.train()
    train_loss_list = []
    train_accuracy_list = []
    
    # 使用tqdm创建进度条
    pbar = tqdm(train_data, desc=f'Epoch {epoch}', ncols=100, position=0, dynamic_ncols=True)
    for data in pbar:
        images, targets = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss_value = vit_loss(outputs, targets)
        loss_value.backward()
        optimizer.step()
        
        train_loss_list.append(loss_value.item())
        prediction = torch.max(outputs, dim=1)[-1]
        train_accuracy = prediction.eq(targets).cpu().float().mean()
        train_accuracy_list.append(train_accuracy)
        
        # 更新进度条信息
        pbar.set_postfix({
            'loss': f'{loss_value.item():.3f}',
            'acc': f'{train_accuracy:.3f}'
        })
    
    train_loss_avg = np.mean(train_loss_list)
    train_accuracy_avg = np.mean(train_accuracy_list)
    
    # 测试过程
    model.eval()
    test_loss_list = []
    test_accuracy_list = []
    
    with torch.no_grad():
        for data in test_data:
            images, targets = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss_value = vit_loss(outputs, targets)
            
            test_loss_list.append(loss_value.item())
            prediction = torch.max(outputs, dim=1)[-1]
            test_accuracy = prediction.eq(targets).cpu().float().mean()
            test_accuracy_list.append(test_accuracy)
    
    test_loss_avg = np.mean(test_loss_list)
    test_accuracy_avg = np.mean(test_accuracy_list)
    
    # 记录训练和测试结果
    log_message = f'Epoch {epoch}: Train Loss: {train_loss_avg:.3f}, Train Acc: {train_accuracy_avg:.3f}, Test Loss: {test_loss_avg:.3f}, Test Acc: {test_accuracy_avg:.3f}'
    with open(log_file, 'a') as f:
        f.write(log_message + '\n')
    print(log_message)
    
    return train_loss_avg, train_accuracy_avg, test_loss_avg, test_accuracy_avg
