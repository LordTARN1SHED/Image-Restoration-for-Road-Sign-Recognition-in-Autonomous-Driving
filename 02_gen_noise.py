# 02_gen_noise.py
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Configure paths
SRC_DIR = Path('./data/gtsrb/GTSRB/Training')
DST_DIR = Path('./data/processed/Noise')

def add_gaussian_noise(image, mean=0, var=0.01):
    """
    Add Gaussian noise
    image: Original image (0-255)
    var: Variance, controls noise intensity
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out

def process():
    if not SRC_DIR.exists():
        print(f"Error: Source data directory {SRC_DIR} not found")
        return

    # Iterate through all class folders
    img_paths = list(SRC_DIR.glob('*/*.ppm')) # GTSRB original format is .ppm
    print(f"Found {len(img_paths)} images, starting to generate noise data...")

    for img_path in tqdm(img_paths):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None: continue

        # Add noise
        noisy_img = add_gaussian_noise(img, var=0.02) # Adjust 'var' to change noise magnitude

        # Maintain directory structure
        relative_path = img_path.relative_to(SRC_DIR)
        save_path = DST_DIR / relative_path
        
        # Create corresponding subfolders
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save (convert to png common format, or keep ppm)
        cv2.imwrite(str(save_path), noisy_img)

    print(f"Processing complete! Noise dataset saved at: {DST_DIR}")

if __name__ == '__main__':
    process()