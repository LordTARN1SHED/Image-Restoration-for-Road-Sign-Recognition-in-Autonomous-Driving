# 15_test_unified.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path

# ================= Configuration Area =================
# 1. Select a test image (Suggest picking one with clear edges, like a speed limit or no entry sign)
SAMPLE_IMG_PATH = Path('./data/gtsrb/GTSRB/Training/00001/00025_00029.ppm')

# 2. Model Paths
UNIFIED_MODEL_PATH = './restoration_unified_resnet.pth'
VGG_MODEL_PATH = './vgg16_baseline.pth' # If missing, VGG scoring might be inaccurate, but the pipeline will still run

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ======================================================

# --- 1. Define Network Structure (Must be exactly consistent with training script 14_train_unified_advanced.py) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.PReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c)
        )
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
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 256)
        )
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResidualBlock(256+128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResidualBlock(128+64, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = ResidualBlock(64+64, 64)
        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        r1 = self.res1(e1)
        p1 = self.pool1(r1)
        r2 = self.res2(p1)
        p2 = self.pool2(r2)
        r3 = self.res3(p2)
        p3 = self.pool3(r3)
        b = self.bottleneck(p3)
        d3 = self.up3(b)
        if d3.size() != r3.size(): d3 = torch.nn.functional.interpolate(d3, size=r3.shape[2:])
        d3 = torch.cat((d3, r3), dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        if d2.size() != r2.size(): d2 = torch.nn.functional.interpolate(d2, size=r2.shape[2:])
        d2 = torch.cat((d2, r2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        if d1.size() != r1.size(): d1 = torch.nn.functional.interpolate(d1, size=r1.shape[2:])
        d1 = torch.cat((d1, r1), dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)

# --- 2. Distortion Utility Functions ---
def make_compound_distortion(image_np):
    """
    Create compound distortion: Simultaneously add Noise + Blur + Fog
    """
    img = image_np.astype(np.float32) / 255.0
    
    # 1. Apply Fog
    intensity = 0.5 
    A = 0.9
    t = 1.0 - intensity
    img = img * t + A * (1 - t)
    
    # 2. Apply Noise
    noise = np.random.normal(0, 0.02 ** 0.5, img.shape)
    img = img + noise
    img = np.clip(img, 0, 1)
    
    # 3. Apply Blur - Needs converting back to uint8 for processing then back again
    temp_img = (img * 255).astype(np.uint8)
    degree = 10
    angle = 45
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    kernel = np.diag(np.ones(degree))
    kernel = cv2.warpAffine(kernel, M, (degree, degree))
    kernel = kernel / degree
    temp_img = cv2.filter2D(temp_img, -1, kernel)
    
    return temp_img # Return uint8 for display and input

# --- 3. Helper Functions ---
def get_vgg_prediction(model, img_tensor):
    """Return VGG predicted class and confidence"""
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

def main():
    print(f"=== Testing Unified ResUNet (Compound Distortion) ===")
    
    # 1. Load Model
    print("Loading Unified restoration model...")
    if not Path(UNIFIED_MODEL_PATH).exists():
        print(f"Error: Model file not found {UNIFIED_MODEL_PATH}, please run 14_train_unified_advanced.py first")
        return
        
    restorer = ResUNet().to(DEVICE)
    restorer.load_state_dict(torch.load(UNIFIED_MODEL_PATH, map_location=DEVICE))
    restorer.eval()
    
    print("Loading VGG16 referee model...")
    vgg = models.vgg16(weights='DEFAULT')
    num_ftrs = vgg.classifier[6].in_features
    vgg.classifier[6] = nn.Linear(num_ftrs, 43)
    if Path(VGG_MODEL_PATH).exists():
        vgg.load_state_dict(torch.load(VGG_MODEL_PATH, map_location=DEVICE))
    vgg = vgg.to(DEVICE)
    vgg.eval()

    # 2. Data Preparation
    # Need two sets of Transforms: One for U-Net (Simple ToTensor), one for VGG (Normalize)
    trans_unet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    trans_vgg = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Read original image
    if not SAMPLE_IMG_PATH.exists():
        print("Error: Image path does not exist")
        return
    
    original_cv = cv2.imread(str(SAMPLE_IMG_PATH))
    original_cv = cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB) # RGB Uint8

    # 3. Generate Data: Clean -> Compound Bad
    bad_cv = make_compound_distortion(original_cv)

    # 4. Execute Restoration (Inference)
    print("Performing blind restoration...")
    # Bad Image -> Tensor -> Device
    bad_pil = Image.fromarray(bad_cv)
    bad_tensor = trans_unet(bad_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        restored_tensor = restorer(bad_tensor)
        
    # Process restoration results for display
    restored_tensor = torch.clamp(restored_tensor, 0, 1)
    restored_img = restored_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    restored_img_uint8 = (restored_img * 255).astype(np.uint8)
    restored_pil = Image.fromarray(restored_img_uint8)

    # 5. VGG Scoring
    # Clean
    clean_pil = Image.fromarray(original_cv)
    pred_c, conf_c = get_vgg_prediction(vgg, trans_vgg(clean_pil).unsqueeze(0).to(DEVICE))
    
    # Bad
    pred_b, conf_b = get_vgg_prediction(vgg, trans_vgg(bad_pil).unsqueeze(0).to(DEVICE))
    
    # Restored
    pred_r, conf_r = get_vgg_prediction(vgg, trans_vgg(restored_pil).unsqueeze(0).to(DEVICE))

    # 6. Visualization Comparison
    print("Generating comparison plot...")
    plt.figure(figsize=(15, 6))

    # Clean
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(original_cv)
    ax1.set_title(f"Original Clean\nVGG Conf: {conf_c:.2f}", fontsize=14, color='green')
    ax1.axis('off')

    # Bad
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(bad_cv)
    ax2.set_title(f"Compound Distorted\n(Noise+Blur+Fog)\nVGG Conf: {conf_b:.2f}", fontsize=14, color='red')
    ax2.axis('off')

    # Restored
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(restored_img_uint8)
    # Determine color based on confidence improvement
    color_r = 'green' if conf_r > 0.8 else ('orange' if conf_r > 0.5 else 'red')
    ax3.set_title(f"Unified Restored\n(Blind Repair)\nVGG Conf: {conf_r:.2f}", fontsize=14, color=color_r)
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig('unified_model_test.png')
    print("Done! Result saved as unified_model_test.png")
    plt.show()

if __name__ == '__main__':
    main()