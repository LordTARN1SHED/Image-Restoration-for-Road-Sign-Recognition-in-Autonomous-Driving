# 16_gen_compound_data.py
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import random

# ================= Configuration =================
SRC_DIR = Path('./data/gtsrb/GTSRB/Training')
DST_DIR = Path('./data/processed/Compound')
# ===============================================

def apply_compound_distortion(image):
    """
    Superposition order: Blur -> Fog -> Noise
    This is the order most consistent with physical logic and highest difficulty
    """
    img = image.astype(np.float32) / 255.0
    
    # 1. Blur (Needs conversion to uint8 for processing)
    temp_img = (img * 255).astype(np.uint8)
    degree = 10; angle = 45
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    k = np.diag(np.ones(degree)); k = cv2.warpAffine(k, M, (degree, degree)) / degree
    temp_img = cv2.filter2D(temp_img, -1, k)
    img = temp_img.astype(np.float32) / 255.0
    
    # 2. Fog
    intensity = 0.5; A = 0.9; t = 1.0 - intensity
    img = img * t + A * (1 - t)
    
    # 3. Noise
    noise = np.random.normal(0, 0.02 ** 0.5, img.shape)
    img = img + noise
    
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def process():
    img_paths = list(SRC_DIR.glob('*/*.ppm'))
    print(f"Starting generation of full compound distortion data (Blur+Fog+Noise)... Total: {len(img_paths)}")

    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        if img is None: continue

        # Generate bad image
        bad_img = apply_compound_distortion(img)

        # Maintain directory structure
        relative_path = img_path.relative_to(SRC_DIR)
        save_path = DST_DIR / relative_path
        
        # Compatibility: Unify saving as png to prevent compression loss
        save_path = save_path.with_suffix('.png')
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), bad_img)

    print(f"Generation complete! Please check: {DST_DIR}")

if __name__ == '__main__':
    process()