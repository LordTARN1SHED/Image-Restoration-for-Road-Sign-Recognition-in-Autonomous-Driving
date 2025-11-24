# 07_train_restoration_advanced.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path

# ================= 配置区域 =================
# 专门为了解决 Blur 掉分问题，我们这次只跑 Blur
TASK_NAME = 'Blur'  

# 参数设置
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0002 # 降低一点学习率，因为Loss变复杂了
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 权重设置 (感知损失的权重)
LAMBDA_PERCEPTUAL = 0.1  # 0.1 是经典经验值

# 路径配置
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')
DISTORTED_DIR = Path(f'./data/processed/{TASK_NAME}')
SAVE_MODEL_PATH = f'./restoration_{TASK_NAME.lower()}.pth'
# ===========================================

print(f"=== 高级训练模式 (Perceptual Loss) ===")
print(f"当前任务: {TASK_NAME}")
print(f"目标: 强制生成锐利边缘以提升 VGG 识别率")

# 1. 定义成对数据集 (保持不变)
class PairedDataset(Dataset):
    def __init__(self, clean_root, distorted_root, transform=None):
        self.clean_root = clean_root
        self.distorted_root = distorted_root
        self.transform = transform
        self.clean_files = sorted(list(clean_root.glob('*/*.ppm')))
        self.data_pairs = []
        for c_path in self.clean_files:
            rel_path = c_path.relative_to(clean_root)
            d_path = distorted_root / rel_path
            if not d_path.exists():
                d_path = d_path.with_suffix('.png')
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

# 2. 定义修复网络 (保持不变)
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(64, 3, 1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        d2 = self.up2(b)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out

# ================= 新增：定义感知损失 (Perceptual Loss) =================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # 加载 VGG16 的特征提取部分
        vgg = models.vgg16(weights='DEFAULT').features
        # 我们只取前 16 层 (包含了前几个卷积块，足以提取纹理和边缘特征)
        self.slice = nn.Sequential()
        for x in range(16):
            self.slice.add_module(str(x), vgg[x])
        
        # 冻结参数，不参与训练
        self.slice.eval()
        for param in self.slice.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # 计算生成图和原图在 VGG 特征空间的距离
        return torch.mean((self.slice(x) - self.slice(y)) ** 2)
# ======================================================================

def train_model():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = PairedDataset(CLEAN_DIR, DISTORTED_DIR, transform=transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = SimpleUNet().to(DEVICE)
    
    # === 修改核心：定义两个 Loss ===
    criterion_pixel = nn.L1Loss() # 使用 L1 而不是 MSE，L1 产生的图像更清晰
    criterion_perceptual = VGGPerceptualLoss().to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("开始训练...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for bad_imgs, clean_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            bad_imgs, clean_imgs = bad_imgs.to(DEVICE), clean_imgs.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(bad_imgs)
            
            # === 计算组合 Loss ===
            loss_pixel = criterion_pixel(outputs, clean_imgs)
            loss_perceptual = criterion_perceptual(outputs, clean_imgs)
            
            # 总 Loss = 像素差异 + 0.1 * VGG特征差异
            loss = loss_pixel + LAMBDA_PERCEPTUAL * loss_perceptual
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.6f}")
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bad_imgs, clean_imgs in val_loader:
                bad_imgs, clean_imgs = bad_imgs.to(DEVICE), clean_imgs.to(DEVICE)
                outputs = model(bad_imgs)
                # 验证时也可以看总Loss
                l_pix = criterion_pixel(outputs, clean_imgs)
                l_perc = criterion_perceptual(outputs, clean_imgs)
                val_loss += (l_pix + LAMBDA_PERCEPTUAL * l_perc).item()
                
        print(f"Epoch {epoch+1} Val Loss: {val_loss/len(val_loader):.6f}")
        
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), SAVE_MODEL_PATH)

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"训练结束！高级模型已保存为 {SAVE_MODEL_PATH}")

if __name__ == '__main__':
    train_model()