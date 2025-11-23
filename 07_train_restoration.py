# 07_train_restoration.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path

# ================= 配置区域 (每次运行修改这里) =================
# 选项: 'Noise', 'Blur', 'Fog'
# 请分别修改这里运行三次！
TASK_NAME = 'Noise'  

# 参数设置
BATCH_SIZE = 32
EPOCHS = 15            # 15轮足够看到明显效果
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')
DISTORTED_DIR = Path(f'./data/processed/{TASK_NAME}')
SAVE_MODEL_PATH = f'./restoration_{TASK_NAME.lower()}.pth'
# ==========================================================

print(f"当前任务: 训练 [{TASK_NAME}] 修复模型")
print(f"输入数据: {DISTORTED_DIR}")
print(f"目标数据: {CLEAN_DIR}")
print(f"模型保存路径: {SAVE_MODEL_PATH}")

# 1. 定义成对数据集 (输入坏图，标签是好图)
class PairedDataset(Dataset):
    def __init__(self, clean_root, distorted_root, transform=None):
        self.clean_root = clean_root
        self.distorted_root = distorted_root
        self.transform = transform
        
        # 寻找所有图片文件
        self.clean_files = sorted(list(clean_root.glob('*/*.ppm')))
        # 确保对应文件夹里也有文件 (只取文件名匹配的)
        self.data_pairs = []
        for c_path in self.clean_files:
            # 构造对应的坏图路径
            rel_path = c_path.relative_to(clean_root)
            d_path = distorted_root / rel_path
            
            # 你的生成脚本有的存为png，有的可能还是ppm，这里做个兼容
            if not d_path.exists():
                d_path = d_path.with_suffix('.png') # 尝试找png
            
            if d_path.exists():
                self.data_pairs.append((d_path, c_path))
        
        print(f"成功匹配图片对: {len(self.data_pairs)} 张")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        d_path, c_path = self.data_pairs[idx]
        
        bad_img = Image.open(d_path).convert('RGB')
        clean_img = Image.open(c_path).convert('RGB')
        
        if self.transform:
            bad_img = self.transform(bad_img)
            clean_img = self.transform(clean_img)
            
        return bad_img, clean_img

# 2. 定义修复网络 (简化版 U-Net)
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Encoder (下采样)
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        
        # Decoder (上采样)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        
        # 输出层
        self.final = nn.Conv2d(64, 3, 1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder (with Skip Connections)
        d2 = self.up2(b)
        d2 = torch.cat((d2, e2), dim=1) # Skip Connection
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1) # Skip Connection
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        return out

def train_model():
    # 数据预处理 (Input/Target 必须大小一致)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载数据
    dataset = PairedDataset(CLEAN_DIR, DISTORTED_DIR, transform=transform)
    
    # 划分训练/验证 (90% 训练, 10% 验证)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 初始化模型
    model = SimpleUNet().to(DEVICE)
    criterion = nn.MSELoss() # 像素级 Loss，让输出尽可能像原图
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("开始训练...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for bad_imgs, clean_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            bad_imgs, clean_imgs = bad_imgs.to(DEVICE), clean_imgs.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(bad_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss (MSE): {avg_loss:.6f}")
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bad_imgs, clean_imgs in val_loader:
                bad_imgs, clean_imgs = bad_imgs.to(DEVICE), clean_imgs.to(DEVICE)
                outputs = model(bad_imgs)
                loss = criterion(outputs, clean_imgs)
                val_loss += loss.item()
        print(f"Epoch {epoch+1} Val Loss: {val_loss/len(val_loader):.6f}")
        
        # 每5轮保存一次检查点，最后也会保存
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), SAVE_MODEL_PATH)

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"训练结束！模型已保存为 {SAVE_MODEL_PATH}")

if __name__ == '__main__':
    train_model()