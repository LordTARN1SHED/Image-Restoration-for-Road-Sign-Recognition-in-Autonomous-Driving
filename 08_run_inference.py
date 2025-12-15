# 08_run_inference.py
import torch
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torchvision import transforms
from PIL import Image

# Import your model definition (Reuse class from 07, or copy here)
# Assuming SimpleUNet is in the same file or pasted here
import torch.nn as nn

# ==========================================
# Must be exactly the same as the model definition in 07_train_restoration.py
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
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')

# Task Configuration
TASKS = ['Noise', 'Blur', 'Fog'] 

def process_task(task_name):
    print(f"\n=== Starting task processing: {task_name} ===")
    
    # 1. Path Settings
    distorted_dir = Path(f'./data/processed/{task_name}')
    restored_dir = Path(f'./data/restored/{task_name}') # Output directory
    model_path = f'./restoration_{task_name.lower()}.pth'
    
    if not os.path.exists(model_path):
        print(f"Warning: Model {model_path} not found, skipping this task.")
        return

    # 2. Load Model
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 3. Prepare Data Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Metrics Statistics
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    # 4. Iterate through all files
    files = list(distorted_dir.glob('*/*.ppm')) + list(distorted_dir.glob('*/*.png'))
    
    for file_path in tqdm(files):
        # Read distorted image
        img_pil = Image.open(file_path).convert('RGB')
        input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        # Post-processing: Convert back to 0-255 image format
        output_tensor = torch.clamp(output_tensor, 0, 1)
        output_img = output_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        output_img = (output_img * 255).astype(np.uint8)
        # RGB -> BGR for OpenCV saving
        output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        # Save image (Maintain original directory structure, important for ImageFolder)
        rel_path = file_path.relative_to(distorted_dir)
        save_path = restored_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Note: We save as png to avoid compression loss
        save_path = save_path.with_suffix('.png')
        cv2.imwrite(str(save_path), output_bgr)
        
        # --- Calculate PSNR/SSIM (For Image Quality Evaluation in Proposal) ---
        # Must load corresponding original image for comparison
        clean_path = CLEAN_DIR / rel_path
        # Original might be ppm, compatibility check
        if not clean_path.exists(): clean_path = clean_path.with_suffix('.ppm')
        
        if clean_path.exists():
            clean_img = cv2.imread(str(clean_path))
            clean_img = cv2.resize(clean_img, (224, 224)) # Ensure consistent size
            
            # Calculate Metrics
            # PSNR
            psnr_val = psnr_metric(clean_img, output_bgr, data_range=255)
            # SSIM (Requires grayscale or multi-channel setting)
            ssim_val = ssim_metric(clean_img, output_bgr, data_range=255, channel_axis=2)
            
            total_psnr += psnr_val
            total_ssim += ssim_val
            count += 1
            
    # Print average metrics for this task
    if count > 0:
        print(f"Task [{task_name}] completed.")
        print(f"Average PSNR: {total_psnr / count:.2f} dB")
        print(f"Average SSIM: {total_ssim / count:.4f}")
    else:
        print("No images processed.")

if __name__ == '__main__':
    for task in TASKS:
        process_task(task)