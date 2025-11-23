# 05_train_baseline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm

# ================= 配置 =================
BATCH_SIZE = 64
EPOCHS = 10           # 跑10轮通常就够了
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = './data/gtsrb/GTSRB/Training'
SAVE_PATH = './vgg16_baseline.pth'
# =======================================

def train():
    print(f"使用设备: {DEVICE}")

    # 1. 数据预处理
    # VGG16 默认输入是 224x224，GTSRB图片大小不一，必须Resize
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 标准均值
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. 加载数据集
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms)
    
    # 划分 80% 训练, 20% 验证
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"训练集数量: {len(train_dataset)}, 验证集数量: {len(val_dataset)}")
    print(f"分类类别数: {len(full_dataset.classes)}")

    # 3. 定义模型 (VGG16)
    # 使用预训练权重 (weights='DEFAULT') 加速收敛
    model = models.vgg16(weights='DEFAULT')
    
    # 冻结前面的特征提取层 (可选，这里我们要微调，所以不完全冻结，但可以锁住前几层)
    # 为了简单且利用你的4090算力，我们直接全参数微调，效果最好
    
    # 修改最后一层全连接层 (原本是1000类，改成43类)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 43)
    
    model = model.to(DEVICE)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 5. 训练循环
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        # --- Training ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / val_total
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # 保存最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"模型已保存，当前最佳验证准确率: {best_acc:.4f}")

    print("训练完成！")

if __name__ == '__main__':
    train()