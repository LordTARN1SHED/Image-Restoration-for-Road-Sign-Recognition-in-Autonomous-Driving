# 09_visualize_result.py
import matplotlib.pyplot as plt
import cv2
import os
import random
from pathlib import Path

# Configuration
TASKS = ['Noise', 'Blur', 'Fog']
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')

def show_comparison():
    # Randomly select an image (e.g., find a specific folder)
    # Find any existing path
    all_files = list(CLEAN_DIR.glob('*/*.ppm'))
    if not all_files: return
    
    # Randomly pick one
    target_file = random.choice(all_files)
    rel_path = target_file.relative_to(CLEAN_DIR)
    
    print(f"Visualizing file: {rel_path}")
    
    plt.figure(figsize=(15, 10))
    
    # First column: Original image
    clean_img = cv2.imread(str(target_file))
    clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(3, 3, 2)
    plt.title("Original Clean Image")
    plt.imshow(clean_img)
    plt.axis('off')
    
    # Iterate through three tasks
    for idx, task in enumerate(TASKS):
        # Distorted image path
        bad_path = Path(f'./data/processed/{task}') / rel_path
        # Some generation scripts might save as png
        if not bad_path.exists(): bad_path = bad_path.with_suffix('.png')
        
        # Restored image path
        restored_path = Path(f'./data/restored/{task}') / rel_path
        restored_path = restored_path.with_suffix('.png') # Restoration script unifies saving as png
        
        if bad_path.exists():
            bad_img = cv2.imread(str(bad_path))
            bad_img = cv2.cvtColor(bad_img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(3, 3, 4 + idx) # Second row
            plt.title(f"{task} (Distorted)")
            plt.imshow(bad_img)
            plt.axis('off')

        if restored_path.exists():
            res_img = cv2.imread(str(restored_path))
            res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(3, 3, 7 + idx) # Third row
            plt.title(f"{task} (Restored)")
            plt.imshow(res_img)
            plt.axis('off')
            
    plt.tight_layout()
    plt.savefig('result_visualization.png')
    print("Comparison plot saved as result_visualization.png")
    plt.show()

if __name__ == '__main__':
    show_comparison()