# 01_download_data.py
import torchvision
import os

def download_gtsrb():
    root_dir = './data'
    print(f"开始下载 GTSRB 数据集到 {root_dir} ...")
    
    # 下载训练集 (我们主要用这个做增强和训练)
    # 这一步会自动下载并解压，可能需要几分钟
    dataset = torchvision.datasets.GTSRB(
        root=root_dir, 
        split='train', 
        download=True
    )
    
    print("下载完成！")
    print(f"数据位置: {os.path.join(root_dir, 'gtsrb')}")

if __name__ == '__main__':
    download_gtsrb()