# 01_download_data.py
import torchvision
import os

def download_gtsrb():
    root_dir = './data'
    print(f"Starting download of GTSRB dataset to {root_dir} ...")
    
    # Download training set (we mainly use this for augmentation and training)
    # This step will automatically download and extract, it may take a few minutes
    dataset = torchvision.datasets.GTSRB(
        root=root_dir, 
        split='train', 
        download=True
    )
    
    print("Download complete!")
    print(f"Data location: {os.path.join(root_dir, 'gtsrb')}")

if __name__ == '__main__':
    download_gtsrb()