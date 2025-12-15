# 17_run_unified_inference.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm

# ================= Configuration =================
DISTORTED_DIR = Path('./data/processed/Compound')
RESTORED_DIR = Path('./data/restored/Compound')
MODEL_PATH = './restoration_unified_resnet.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 # Batch processing is faster
# ===============================================

# --- ResUNet Definition (Must match training) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.PReLU(), nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c))
        self.shortcut = nn.Sequential()
        if in_c != out_c: self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1), nn.BatchNorm2d(out_c))
    def forward(self, x): return torch.nn.functional.relu(self.conv_block(x) + self.shortcut(x))

class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.PReLU())
        self.res1 = ResidualBlock(64, 64); self.pool1 = nn.MaxPool2d(2, 2)
        self.res2 = ResidualBlock(64, 128); self.pool2 = nn.MaxPool2d(2, 2)
        self.res3 = ResidualBlock(128, 256); self.pool3 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(ResidualBlock(256, 512), ResidualBlock(512, 512), ResidualBlock(512, 256))
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec3 = ResidualBlock(384, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.dec2 = ResidualBlock(192, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2); self.dec1 = ResidualBlock(128, 64)
        self.final = nn.Conv2d(64, 3, 1)
    def forward(self, x):
        e1 = self.enc1(x); r1 = self.res1(e1); p1 = self.pool1(r1)
        r2 = self.res2(p1); p2 = self.pool2(r2)
        r3 = self.res3(p2); p3 = self.pool3(r3)
        b = self.bottleneck(p3)
        d3 = self.up3(b); 
        if d3.size()!=r3.size(): d3=torch.nn.functional.interpolate(d3,size=r3.shape[2:])
        d3 = torch.cat((d3, r3), dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); 
        if d2.size()!=r2.size(): d2=torch.nn.functional.interpolate(d2,size=r2.shape[2:])
        d2 = torch.cat((d2, r2), dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); 
        if d1.size()!=r1.size(): d1=torch.nn.functional.interpolate(d1,size=r1.shape[2:])
        d1 = torch.cat((d1, r1), dim=1); d1 = self.dec1(d1)
        return self.final(d1)

def run_inference():
    print("Loading unified restoration model...")
    model = ResUNet().to(DEVICE)
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found {MODEL_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    # Get all files
    files = list(DISTORTED_DIR.glob('*/*.png'))
    print(f"Starting restoration of {len(files)} images...")
    
    # Batch processing logic
    for i in tqdm(range(0, len(files), BATCH_SIZE)):
        batch_files = files[i : i+BATCH_SIZE]
        
        # Prepare Batch
        inputs = []
        for p in batch_files:
            img = Image.open(p).convert('RGB')
            inputs.append(transform(img))
        
        input_tensor = torch.stack(inputs).to(DEVICE)
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
            output_tensor = torch.clamp(output_tensor, 0, 1)
        
        # Save Batch
        for idx, file_path in enumerate(batch_files):
            # Tensor -> Numpy -> BGR
            out_img = output_tensor[idx].cpu().permute(1, 2, 0).numpy()
            out_img = (out_img * 255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            
            # Path calculation
            rel_path = file_path.relative_to(DISTORTED_DIR)
            save_path = RESTORED_DIR / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), out_bgr)

    print(f"Restoration complete! Please check: {RESTORED_DIR}")

if __name__ == '__main__':
    run_inference()