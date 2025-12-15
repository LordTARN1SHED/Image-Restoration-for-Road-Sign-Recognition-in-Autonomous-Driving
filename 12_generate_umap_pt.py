# 12_generate_umap_pt.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import random
from pathlib import Path
from tqdm import tqdm

# ================= Configuration Area =================
# Sample count: How many images per mode? (Suggested 50-100, too many makes it cluttered, too few lacks representation)
SAMPLES_PER_MODE = 100 

# Base path
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')
# Define 7 modes and their paths
DATA_MODES = {
    'Clean':            CLEAN_DIR,
    'Noise (Bad)':      Path('./data/processed/Noise'),
    'Noise (Restored)': Path('./data/restored/Noise'),
    'Blur (Bad)':       Path('./data/processed/Blur'),
    'Blur (Restored)':  Path('./data/restored/Blur'),
    'Fog (Bad)':        Path('./data/processed/Fog'),
    'Fog (Restored)':   Path('./data/restored/Fog')
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PT_PATH = 'umap_embeddings.pt'
OUTPUT_IMG_PATH = 'umap_visualization.png'

# ======================================================

def get_vgg_features(model, img_tensor):
    """
    Extract final output from VGG features module
    Output Shape: [1, 512, 7, 7]
    """
    with torch.no_grad():
        features = model.features(img_tensor)
    return features

def process_features_for_umap(feature_tensor):
    """
    Key step: Dimensionality transformation
    Input:  [1, 512, 7, 7]
    Output: [1, 512] (numpy array)
    Strategy: Global Average Pooling (GAP)
    """
    # 1. Average over spatial dimensions 7x7 (dim=2, 3)
    # Result becomes [1, 512]
    pooled = torch.mean(feature_tensor, dim=[2, 3])
    
    # 2. Convert to Numpy
    return pooled.cpu().numpy()

def main():
    print("1. Loading VGG16 model...")
    # We only need the feature extractor, not the classifier
    vgg = models.vgg16(weights='DEFAULT').to(DEVICE)
    vgg.eval()

    # Preprocessing (Must match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare containers
    all_features = [] # Store X: (N_total, 512)
    all_labels = []   # Store y: (N_total,) -> Corresponding mode names

    # 2. Iterate through 7 modes to extract features
    # To ensure fair comparison, we should try to pick images with the same names
    # First get list of all Clean image filenames
    all_files = list(CLEAN_DIR.glob('*/*.ppm'))
    if len(all_files) > SAMPLES_PER_MODE:
        selected_files = random.sample(all_files, SAMPLES_PER_MODE)
    else:
        selected_files = all_files

    print(f"Sampling {len(selected_files)} images per mode, total 7 modes...")

    for mode_name, dir_path in DATA_MODES.items():
        print(f"Processing: {mode_name} ...")
        
        valid_count = 0
        for clean_path in tqdm(selected_files):
            # Construct relative path to find corresponding Bad/Restored files
            rel_path = clean_path.relative_to(CLEAN_DIR)
            
            # Determine file path for current mode
            if mode_name == 'Clean':
                target_path = clean_path
            else:
                target_path = dir_path / rel_path
                # Compatible with png
                if not target_path.exists():
                    target_path = target_path.with_suffix('.png')
            
            if not target_path.exists():
                continue

            # Read and preprocess
            try:
                img = Image.open(target_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(DEVICE) # [1, 3, 224, 224]
                
                # Extract features [1, 512, 7, 7]
                raw_feat = get_vgg_features(vgg, tensor)
                
                # Flatten to [1, 512]
                flat_feat = process_features_for_umap(raw_feat)
                
                all_features.append(flat_feat)
                all_labels.append(mode_name)
                valid_count += 1
            except Exception as e:
                print(f"Error processing {target_path}: {e}")

    # 3. Run UMAP
    print("\nRunning UMAP dimensionality reduction (this may take a few seconds)...")
    # Stack into a large matrix: (Total_Samples, 512)
    X = np.vstack(all_features)
    
    # UMAP Configuration
    reducer = umap.UMAP(
        n_neighbors=15,    # number of neighbors, larger values focus more on global structure
        min_dist=0.1,      # minimum distance between points, smaller values result in tighter clusters
        n_components=2,    # reduce to 2D
        metric='cosine',   # use cosine similarity (usually better than Euclidean in high-dimensional space)
        random_state=42    # fix random seed
    )
    
    embedding = reducer.fit_transform(X) # Output: (Total_Samples, 2)
    
    print(f"Dimensionality reduction complete! Input shape: {X.shape}, Output shape: {embedding.shape}")

    # 4. Save as .pt file (as requested)
    print(f"Saving data to {OUTPUT_PT_PATH} ...")
    data_to_save = {
        'embeddings': torch.tensor(embedding), # UMAP coordinates (x, y)
        'labels': all_labels,                  # List of corresponding class names
        'original_features': torch.tensor(X)   # Original 512-dim vectors (for backup)
    }
    torch.save(data_to_save, OUTPUT_PT_PATH)

    # 5. Plotting visualization
    print(f"Generating visualization chart {OUTPUT_IMG_PATH} ...")
    plt.figure(figsize=(12, 10))
    
    # Use Seaborn for plotting because it handles string labels conveniently
    sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1], 
        hue=all_labels, 
        palette="tab10", # Color palette
        s=60,            # Dot size
        alpha=0.7        # Transparency
    )
    
    plt.title('UMAP Projection of VGG16 Features (Layer: features.30)', fontsize=15)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # Legend outside
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH)
    plt.show()

    print("All done!")

if __name__ == '__main__':
    main()