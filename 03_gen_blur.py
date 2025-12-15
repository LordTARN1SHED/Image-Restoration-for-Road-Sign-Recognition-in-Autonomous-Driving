# 03_gen_blur.py
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

SRC_DIR = Path('./data/gtsrb/GTSRB/Training')
DST_DIR = Path('./data/processed/Blur')

def apply_motion_blur(image, degree=10, angle=45):
    """
    Add motion blur
    degree: Kernel size (motion distance)
    angle: Motion angle
    """
    image = np.array(image)
    
    # Generate blur kernel
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    
    # Convolution
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    
    # Maintain original data type
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    return np.array(blurred, dtype=np.uint8)

def process():
    img_paths = list(SRC_DIR.glob('*/*.ppm'))
    print(f"Found {len(img_paths)} images, starting to generate motion blur data...")

    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        if img is None: continue

        # Add blur
        blur_img = apply_motion_blur(img, degree=12, angle=45) 

        relative_path = img_path.relative_to(SRC_DIR)
        save_path = DST_DIR / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), blur_img)

    print(f"Processing complete! Blur dataset saved at: {DST_DIR}")

if __name__ == '__main__':
    process()