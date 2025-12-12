# 14_train_unified_advanced.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import random
from pathlib import Path
from tqdm import tqdm

# ================= 高级配置 =================
# 这个模型要处理所有情况，所以Epoch要多一点，BatchSize看显存调整
BATCH_SIZE = 16 
EPOCHS = 25 
LEARNING_RATE = 0.0002 # 较小的LR，配合AdamW，训练更稳定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH = './restoration_unified_resnet.pth'
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')

# 混合畸变的概率配置
PROB_NOISE = 0.5
PROB_BLUR = 0.5
PROB_FOG = 0.5
# ===========================================

# --- 1. 动态畸变生成器 (核心升级) ---
# 这里的代码直接复用了之前的逻辑，但变成了随机触发
def apply_random_distortions(image_np):
    """
    输入: 0-255 RGB Numpy
    输出: 0-255 RGB Numpy (混合畸变)
    """
    out = image_np.astype(np.float32) / 255.0
    
    # 随机施加 Fog
    if random.random() < PROB_FOG:
        intensity = random.uniform(0.3, 0.7) # 混合时不要太浓，否则信息丢失太多
        A = 0.9
        t = 1.0 - intensity * random.uniform(0.8, 1.2)
        out = out * t + A * (1 - t)
    
    # 随机施加 Noise (必须Clip，否则下一层会溢出)
    if random.random() < PROB_NOISE:
        var = random.uniform(0.01, 0.03) 
        noise = np.random.normal(0, var ** 0.5, out.shape)
        out = out + noise
        
    # 随机施加 Blur (转回 uint8 处理再转回来，方便用cv2)
    if random.random() < PROB_BLUR:
        temp_img = np.clip(out * 255, 0, 255).astype(np.uint8)
        degree = random.randint(5, 15)
        angle = random.randint(0, 360)
        if degree > 1:
            M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
            kernel = np.diag(np.ones(degree))
            kernel = cv2.warpAffine(kernel, M, (degree, degree))
            kernel = kernel / degree
            temp_img = cv2.filter2D(temp_img, -1, kernel)
        out = temp_img.astype(np.float32) / 255.0

    return np.clip(out * 255, 0, 255).astype(np.uint8)

class DynamicDistortionDataset(Dataset):
    def __init__(self, clean_root, transform=None):
        self.clean_files = sorted(list(clean_root.glob('*/*.ppm')))
        self.transform = transform
        print(f"载入训练数据: {len(self.clean_files)} 张图片")

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        c_path = self.clean_files[idx]
        
        # 读取 Clean
        clean_img_cv = cv2.imread(str(c_path))
        clean_img_cv = cv2.cvtColor(clean_img_cv, cv2.COLOR_BGR2RGB)
        
        # 实时生成 Bad (Dynamic Generation)
        bad_img_cv = apply_random_distortions(clean_img_cv)
        
        # 转 PIL 以便使用 torchvision transforms
        clean_pil = Image.fromarray(clean_img_cv)
        bad_pil = Image.fromarray(bad_img_cv)
        
        if self.transform:
            clean_tensor = self.transform(clean_pil)
            bad_tensor = self.transform(bad_pil)
            
        return bad_tensor, clean_tensor

# --- 2. 模型升级: ResUNet (Residual U-Net) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.PReLU(), # PReLU 比 ReLU 更好
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        # 如果通道数变了，Shortcut也要变
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        return torch.nn.functional.relu(self.conv_block(x) + self.shortcut(x))

class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.PReLU())
        self.res1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.res2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.res3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck (更深)
        self.bottleneck = nn.Sequential(
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 256)
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResidualBlock(256+128, 128) # Skip connection concat size
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResidualBlock(128+64, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = ResidualBlock(64+64, 64)
        
        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        # Enc
        e1 = self.enc1(x)
        r1 = self.res1(e1)
        p1 = self.pool1(r1)
        
        r2 = self.res2(p1)
        p2 = self.pool2(r2)
        
        r3 = self.res3(p2)
        p3 = self.pool3(r3)
        
        # Bottle
        b = self.bottleneck(p3)
        
        # Dec
        d3 = self.up3(b)
        # Skip connection 这里的尺寸可能会因为Padding有细微差别，插值对其
        if d3.size() != r3.size():
            d3 = torch.nn.functional.interpolate(d3, size=r3.shape[2:])
        d3 = torch.cat((d3, r3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.size() != r2.size():
            d2 = torch.nn.functional.interpolate(d2, size=r2.shape[2:])
        d2 = torch.cat((d2, r2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.size() != r1.size():
            d1 = torch.nn.functional.interpolate(d1, size=r1.shape[2:])
        d1 = torch.cat((d1, r1), dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

# --- 3. Loss 升级: Perceptual Loss (复用) ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights='DEFAULT').features[:16]
        self.slice = vgg.eval()
        for p in self.slice.parameters(): p.requires_grad = False
    def forward(self, x, y):
        return torch.mean((self.slice(x) - self.slice(y)) ** 2)

# ================= 训练循环 =================
def train():
    print("=== 开始训练 Unified ResUNet (Mixed Distortion) ===")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # 使用动态数据集
    dataset = DynamicDistortionDataset(CLEAN_DIR, transform=transform)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8) # 多线程生成很重要
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = ResUNet().to(DEVICE)
    
    # 组合损失: L1 (像素准) + Perceptual (视觉准)
    crit_l1 = nn.L1Loss()
    crit_perc = VGGPerceptualLoss().to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        run_loss = 0.0
        
        # 这里的 tqdm 会稍微慢一点，因为 CPU 在实时生成扭曲图
        for bad, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            bad, clean = bad.to(DEVICE), clean.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(bad)
            
            l_pix = crit_l1(out, clean)
            l_perc = crit_perc(out, clean)
            
            # 0.1 的权重给 Perceptual 是为了平衡量级
            loss = l_pix + 0.1 * l_perc
            
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        
        scheduler.step()
        avg_loss = run_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.5f}")
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bad, clean in val_loader:
                bad, clean = bad.to(DEVICE), clean.to(DEVICE)
                out = model(bad)
                l = crit_l1(out, clean) + 0.1 * crit_perc(out, clean)
                val_loss += l.item()
        
        val_avg = val_loss / len(val_loader)
        print(f"Val Loss: {val_avg:.5f}")
        
        if val_avg < best_loss:
            best_loss = val_avg
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print("模型已保存 (Best Val)")

if __name__ == '__main__':
    train()