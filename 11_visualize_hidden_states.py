# 11_visualize_hidden_states.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

# ================= Configuration =================
# Select a typical image (e.g., speed limit sign with sharp edges)
# Please replace with the relative path of an image existing locally
SAMPLE_IMG_REL_PATH = '00001/00025_00029.ppm' 

# Base paths
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')
BAD_DIRS = {
    'Noise': Path('./data/processed/Noise'),
    'Blur':  Path('./data/processed/Blur'),
    'Fog':   Path('./data/processed/Fog')
}
RESTORED_DIRS = {
    'Noise': Path('./data/restored/Noise'),
    'Blur':  Path('./data/restored/Blur'),
    'Fog':   Path('./data/restored/Fog')
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_vgg_feature_maps(model, img_tensor, layer_index=6):
    """
    Extract feature maps from a specific layer of VGG16
    layer_index=0 -> 1st conv layer (Conv1_1)
    layer_index=2 -> 2nd conv layer (Conv1_2)
    layer_index=5 -> 3rd conv layer (Conv2_1) ... and so on
    """
    # Extract the first N layers of VGG
    feature_extractor = model.features[:layer_index+1]
    
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    
    return features

def plot_heatmap(features):
    """
    Average multi-channel feature maps to a single-channel heatmap for visualization
    """
    # features shape: [1, 64, 224, 224]
    # Average over Channel dimension -> [1, 224, 224]
    heatmap = torch.mean(features, dim=1).squeeze().cpu().numpy()
    
    # Normalize to 0-1 for plotting
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap

def main():
    # 1. Prepare model (We don't need our fine-tuned weights, just ImageNet pretrained weights for feature extraction)
    # Because we want to see how VGG's "general" visual capability is affected
    print("Loading VGG16 model...")
    vgg = models.vgg16(weights='DEFAULT').to(DEVICE)
    vgg.eval()
    
    # 2. Prepare preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalization is optional here; used to see raw activations intuitively or keep consistency
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Find file
    clean_path = CLEAN_DIR / SAMPLE_IMG_REL_PATH
    if not clean_path.exists():
        print(f"Error: File {clean_path} not found, please modify SAMPLE_IMG_REL_PATH in the script")
        return

    # 4. Start plotting
    # We compare three distortions, one per row
    # Each row shows: Clean Feature | Bad Feature | Restored Feature
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    # Column titles
    cols = ["Input Image (Bad/Restored)", "Clean Features", "Distorted Features", "Restored Features"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold')

    tasks = ['Noise', 'Blur', 'Fog']
    
    # Which VGG layer to view?
    # Layer 2 (after ReLU) usually shows clear edges
    TARGET_LAYER = 2 

    print(f"Extracting Hidden States from VGG16 layer {TARGET_LAYER}...")

    for i, task in enumerate(tasks):
        # Construct paths
        bad_path = BAD_DIRS[task] / SAMPLE_IMG_REL_PATH
        if not bad_path.exists(): bad_path = bad_path.with_suffix('.png')
        
        restored_path = RESTORED_DIRS[task] / SAMPLE_IMG_REL_PATH
        restored_path = restored_path.with_suffix('.png')

        # Read images
        img_clean = Image.open(clean_path).convert('RGB')
        img_bad = Image.open(bad_path).convert('RGB')
        img_res = Image.open(restored_path).convert('RGB')

        # Convert to Tensor
        t_clean = transform(img_clean).unsqueeze(0).to(DEVICE)
        t_bad = transform(img_bad).unsqueeze(0).to(DEVICE)
        t_res = transform(img_res).unsqueeze(0).to(DEVICE)

        # Extract features
        f_clean = get_vgg_feature_maps(vgg, t_clean, TARGET_LAYER)
        f_bad = get_vgg_feature_maps(vgg, t_bad, TARGET_LAYER)
        f_res = get_vgg_feature_maps(vgg, t_res, TARGET_LAYER)

        # Convert to heatmap
        h_clean = plot_heatmap(f_clean)
        h_bad = plot_heatmap(f_bad)
        h_res = plot_heatmap(f_res)

        # --- Plotting ---
        # First column: Show the restored image (RGB)
        axes[i, 0].imshow(img_res)
        axes[i, 0].set_ylabel(task, fontsize=14, fontweight='bold')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # Second column: Clean Features
        axes[i, 1].imshow(h_clean, cmap='viridis')
        axes[i, 1].axis('off')

        # Third column: Bad Image Features
        axes[i, 2].imshow(h_bad, cmap='viridis')
        axes[i, 2].axis('off')

        # Fourth column: Restored Image Features
        axes[i, 3].imshow(h_res, cmap='viridis')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('hidden_state_visualization.png')
    print("Visualization complete! Saved as hidden_state_visualization.png")
    plt.show()

if __name__ == '__main__':
    main()