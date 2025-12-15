# 04_gen_fog.py
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import random

SRC_DIR = Path('./data/gtsrb/GTSRB/Training')
DST_DIR = Path('./data/processed/Fog')

def add_fog(image, fog_intensity=0.8):
    """
    Add fog effect (simple linear blending)
    fog_intensity: 0.0 (no fog) to 1.0 (full white)
    """
    image = np.array(image) / 255.0
    
    # Define atmospheric light A (set to pure white here, slightly gray might be more realistic)
    A = 0.9 
    
    # Generate transmission t (simplified as global uniform fog here; advanced methods use depth maps for non-uniform t)
    # We add a bit of randomness so the fog looks slightly different for each image
    t = 1.0 - fog_intensity * random.uniform(0.8, 1.2)
    t = np.clip(t, 0.1, 0.9) # Limit range to prevent full black or white
    
    # Physical model: I = J*t + A*(1-t)
    fog_img = image * t + A * (1 - t)
    
    fog_img = np.clip(fog_img * 255, 0, 255).astype(np.uint8)
    return fog_img

def process():
    img_paths = list(SRC_DIR.glob('*/*.ppm'))
    print(f"Found {len(img_paths)} images, starting to generate fog data...")

    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        if img is None: continue

        # Add fog
        fog_img = add_fog(img, fog_intensity=0.8) 

        relative_path = img_path.relative_to(SRC_DIR)
        save_path = DST_DIR / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), fog_img)

    print(f"Processing complete! Fog dataset saved at: {DST_DIR}")

if __name__ == '__main__':
    process()