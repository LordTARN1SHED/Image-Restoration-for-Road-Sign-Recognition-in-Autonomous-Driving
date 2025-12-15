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

# ================= Advanced Configuration =================
# Since this model handles all scenarios, epochs should be slightly higher. Adjust BatchSize based on VRAM.
BATCH_SIZE = 16 
EPOCHS = 25 
LEARNING_RATE = 0.0002 # Smaller LR with AdamW for more stable training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH = './restoration_unified_resnet.pth'
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')

# Probability configuration for mixed distortions
PROB_NOISE = 0.5
PROB_BLUR = 0.5
PROB_FOG = 0.5
# ==========================================================

# --- 1. Dynamic Distortion Generator (Core Upgrade) ---
# The logic is reused from before but triggered randomly
def apply_random_distortions(image_np):
    """
    Input: 0-255 RGB Numpy
    Output: 0-255 RGB Numpy (Mixed Distortion)
    """
    out = image_np.astype(np.float32) / 255.0
    
    # Randomly apply Fog
    if random.random() < PROB_FOG:
        intensity = random.uniform(0.3, 0.7) # Not too thick when mixing, otherwise too much info is lost
        A = 0.9
        t = 1.0 - intensity * random.uniform(0.8, 1.2)
        out = out * t + A * (1 - t)
    
    # Randomly apply Noise (Must clip, otherwise next layer overflows)
    if random.random() < PROB_NOISE:
        var = random.uniform(0.01, 0.03) 
        noise = np.random.normal(0, var ** 0.5, out.shape)
        out = out + noise
        
    # Randomly apply Blur (Convert back to uint8 for cv2 processing, then convert back)
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
        print(f"Loading training data: {len(self.clean_files)} images")

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        c_path = self.clean_files[idx]
        
        # Read Clean Image
        clean_img_cv = cv2.imread(str(c_path))
        clean_img_cv = cv2.cvtColor(clean_img_cv, cv2.COLOR_BGR2RGB)
        
        # Real-time Generation of Bad Image (Dynamic Generation)
        bad_img_cv = apply_random_distortions(clean_img_cv)
        
        # Convert to PIL for torchvision transforms
        clean_pil = Image.fromarray(clean_img_cv)
        bad_pil = Image.fromarray(bad_img_cv)
        
        if self.transform:
            clean_tensor = self.transform(clean_pil)
            bad_tensor = self.transform(bad_pil)
            
        return bad_tensor, clean_tensor

# --- 2. Model Upgrade: ResUNet (Residual U-Net) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.PReLU(), # PReLU is better than ReLU
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        # If channel number changes, Shortcut must change too
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
        
        # Bottleneck (Deeper)
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
        # Skip connection dimensions might differ slightly due to padding, align via interpolation
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

# --- 3. Loss Upgrade: Perceptual Loss (Reused) ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights='DEFAULT').features[:16]
        self.slice = vgg.eval()
        for p in self.slice.parameters(): p.requires_grad = False
    def forward(self, x, y):
        return torch.mean((self.slice(x) - self.slice(y)) ** 2)

# ================= Training Loop =================
def train():
    print("=== Starting Training Unified ResUNet (Mixed Distortion) ===")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Use dynamic dataset
    dataset = DynamicDistortionDataset(CLEAN_DIR, transform=transform)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8) # Multi-threading is important for generation
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = ResUNet().to(DEVICE)
    
    # Combined Loss: L1 (Pixel accuracy) + Perceptual (Visual accuracy)
    crit_l1 = nn.L1Loss()
    crit_perc = VGGPerceptualLoss().to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        run_loss = 0.0
        
        # tqdm here might be slightly slower because CPU is generating distorted images in real-time
        for bad, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            bad, clean = bad.to(DEVICE), clean.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(bad)
            
            l_pix = crit_l1(out, clean)
            l_perc = crit_perc(out, clean)
            
            # 0.1 weight for Perceptual is to balance the magnitude
            loss = l_pix + 0.1 * l_perc
            
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        
        scheduler.step()
        avg_loss = run_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.5f}")
        
        # Validation
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
            print("Model saved (Best Val)")

if __name__ == '__main__':
    train()