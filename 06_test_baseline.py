# 06_test_baseline.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# ================= 配置 =================
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './vgg16_baseline.pth'

# 测试数据集路径列表
TEST_DIRS = {
    "Clean (Original)": './data/gtsrb/GTSRB/Training',
    "Noisy":            './data/processed/Noise',
    "Blurred":          './data/processed/Blur',
    "Foggy":            './data/processed/Fog'
}
# =======================================

def evaluate_model(model, data_dir, name):
    """
    评估模型在指定目录下的准确率
    """
    if not os.path.exists(data_dir):
        print(f"跳过 {name}: 路径不存在 {data_dir}")
        return

    # 必须和训练时一样的预处理
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    # 这里为了快一点，如果是为了验证趋势，我们可以只取一部分数据，或者跑全量
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    correct = 0
    total = 0
    
    model.eval()
    print(f"\n正在测试: {name} ...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"结果 [{name}] 准确率: {acc*100:.2f}%")
    return acc

def main():
    print(f"使用设备: {DEVICE}")

    # 1. 加载模型架构
    model = models.vgg16(weights=None) # 这里不需要下载权重了，因为我们要加载自己训练的pth
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 43) # 必须和训练时结构一致
    
    # 2. 加载权重
    if not os.path.exists(MODEL_PATH):
        print("错误：找不到模型文件，请先运行 05_train_baseline.py")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICE)
    print("模型加载成功！")

    # 3. 依次测试
    results = {}
    for name, path in TEST_DIRS.items():
        acc = evaluate_model(model, path, name)
        if acc is not None:
            results[name] = acc

    # 4. 打印最终汇总
    print("\n" + "="*30)
    print("最终测试报告 (Baseline 1)")
    print("="*30)
    print(f"{'Dataset':<20} | {'Accuracy':<10}")
    print("-" * 32)
    for name, acc in results.items():
        print(f"{name:<20} | {acc*100:.2f}%")
    print("="*30)

if __name__ == '__main__':
    main()