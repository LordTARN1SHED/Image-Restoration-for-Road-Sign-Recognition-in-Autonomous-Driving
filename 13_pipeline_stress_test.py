# 13_pipeline_stress_test_multi.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path
import os

# ================= Configuration Area =================
NUM_SAMPLES = 10  # How many images to test?
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')
OUTPUT_DIR = Path('./pipeline_results') # Result save directory

# Model Paths
MODEL_PATHS = {
    'Noise': './restoration_noise.pth',
    'Blur':  './restoration_blur.pth',
    'Fog':   './restoration_fog.pth'
}

# Distortion superposition order: Blur -> Fog -> Noise
# Restoration execution order: Noise -> Fog -> Blur
RESTORATION_ORDER = ['Noise', 'Fog', 'Blur']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ======================================================

# --- 1. Define Distortion Functions ---
def add_noise(image):
    img = image / 255.0
    noise = np.random.normal(0, 0.01 ** 0.5, img.shape) 
    out = img + noise
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)

def add_blur(image):
    degree = 5 
    angle = 45
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    return blurred

def add_fog(image):
    fog_intensity = 0.1 
    img = image / 255.0
    A = 0.9
    t = 1.0 - fog_intensity
    fog_img = img * t + A * (1 - t)
    return np.clip(fog_img * 255, 0, 255).astype(np.uint8)

# --- 2. Define Model Structure ---
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

def get_vgg_prediction(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Prepare Models
    print("Loading models...")
    restoration_models = {}
    for name in MODEL_PATHS:
        if Path(MODEL_PATHS[name]).exists():
            net = SimpleUNet().to(DEVICE)
            net.load_state_dict(torch.load(MODEL_PATHS[name], map_location=DEVICE))
            net.eval()
            restoration_models[name] = net
        else:
            print(f"Warning: Model not found {MODEL_PATHS[name]}")
    
    vgg = models.vgg16(weights='DEFAULT')
    num_ftrs = vgg.classifier[6].in_features
    vgg.classifier[6] = nn.Linear(num_ftrs, 43)
    if Path('./vgg16_baseline.pth').exists():
        vgg.load_state_dict(torch.load('./vgg16_baseline.pth', map_location=DEVICE))
    vgg = vgg.to(DEVICE)
    vgg.eval()

    preprocess_vgg = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    preprocess_unet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 2. Randomly Sample Images
    all_files = list(CLEAN_DIR.glob('*/*.ppm'))
    if len(all_files) < NUM_SAMPLES:
        selected_files = all_files
    else:
        selected_files = random.sample(all_files, NUM_SAMPLES)
    
    print(f"Selected {len(selected_files)} images for testing...\n")

    # Statistics containers
    stats_clean_conf = []
    stats_bad_conf = []
    stats_restored_conf = []

    # 3. Loop through each image
    for idx, img_path in enumerate(selected_files):
        print(f"[{idx+1}/{NUM_SAMPLES}] Processing: {img_path.name}")
        
        # Read original image
        original_cv = cv2.imread(str(img_path))
        original_cv = cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB)
        
        history_images = [original_cv]
        history_titles = ["Original"]
        
        # --- Phase 1: Superimpose Distortions (Blur -> Fog -> Noise) ---
        current_img = original_cv.copy()
        
        # Blur
        current_img = add_blur(current_img)
        history_images.append(current_img)
        history_titles.append("+ Blur")
        
        # Fog
        current_img = add_fog(current_img)
        history_images.append(current_img)
        history_titles.append("+ Fog")
        
        # Noise
        current_img = add_noise(current_img)
        history_images.append(current_img)
        history_titles.append("+ Noise (Input)")
        
        # Backup bad image for VGG scoring
        bad_img_for_stats = current_img.copy()

        # --- Phase 2: Cascade Restoration (Noise -> Fog -> Blur) ---
        tensor_img = preprocess_unet(Image.fromarray(current_img)).unsqueeze(0).to(DEVICE)
        
        for model_name in RESTORATION_ORDER:
            if model_name in restoration_models:
                net = restoration_models[model_name]
                with torch.no_grad():
                    tensor_img = net(tensor_img)
                    
                    # Visualization saving
                    vis_tensor = torch.clamp(tensor_img, 0, 1)
                    vis_img = vis_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
                    vis_img = (vis_img * 255).astype(np.uint8)
                    
                    history_images.append(vis_img)
                    history_titles.append(f"After {model_name}")

        # --- Phase 3: VGG Scoring and Saving Images ---
        plt.figure(figsize=(20, 8))
        
        # Temp variables to record three key scores for this image
        conf_c, conf_b, conf_r = 0, 0, 0

        for i, (img, title) in enumerate(zip(history_images, history_titles)):
            pil_img = Image.fromarray(img)
            vgg_input = preprocess_vgg(pil_img).unsqueeze(0).to(DEVICE)
            
            pred_cls, conf = get_vgg_prediction(vgg, vgg_input)
            
            # Record confidence at key nodes
            if i == 0: conf_c = conf          # Original
            if i == 3: conf_b = conf          # + Noise (Final Bad)
            if i == 6: conf_r = conf          # Final Restored
            
            ax = plt.subplot(2, 4, i + 1)
            ax.imshow(img)
            color = 'green' if conf > 0.8 else ('orange' if conf > 0.5 else 'red')
            ax.set_title(f"{title}\nPred: {pred_cls} | Conf: {conf:.2f}", color=color, fontsize=12, fontweight='bold')
            ax.axis('off')

        # Save image
        save_path = OUTPUT_DIR / f'pipeline_sample_{idx+1}.png'
        plt.tight_layout()
        plt.savefig(str(save_path))
        plt.close() # Close figure to free memory
        
        # Add to statistics
        stats_clean_conf.append(conf_c)
        stats_bad_conf.append(conf_b)
        stats_restored_conf.append(conf_r)

    # 4. Print Final Statistical Report
    avg_clean = sum(stats_clean_conf) / len(stats_clean_conf)
    avg_bad = sum(stats_bad_conf) / len(stats_bad_conf)
    avg_restored = sum(stats_restored_conf) / len(stats_restored_conf)

    print("\n" + "="*40)
    print(f"Final Test Report (Total {NUM_SAMPLES} images)")
    print("="*40)
    print(f"{'Stage':<20} | {'Avg Confidence':<15}")
    print("-" * 38)
    print(f"{'Original (Clean)':<20} | {avg_clean:.4f}")
    print(f"{'Distorted (Input)':<20} | {avg_bad:.4f}")
    print(f"{'Restored (Output)':<20} | {avg_restored:.4f}")
    print("="*40)
    print(f"All result images saved in: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()