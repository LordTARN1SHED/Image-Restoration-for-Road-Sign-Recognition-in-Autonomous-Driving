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
    添加雾霾效果 (简单的线性混合)
    fog_intensity: 0.0 (无雾) 到 1.0 (全白)
    """
    image = np.array(image) / 255.0
    
    # 定义大气光 A (这里设为纯白，稍微带点灰会更真实)
    A = 0.9 
    
    # 生成透射率 t (这里简化为全局均匀雾，高级做法是用深度图生成不均匀t)
    # 我们加一点随机性让每张图的雾看起来稍微不同
    t = 1.0 - fog_intensity * random.uniform(0.8, 1.2)
    t = np.clip(t, 0.1, 0.9) # 限制范围防止全黑或全白
    
    # 物理模型: I = J*t + A*(1-t)
    fog_img = image * t + A * (1 - t)
    
    fog_img = np.clip(fog_img * 255, 0, 255).astype(np.uint8)
    return fog_img

def process():
    img_paths = list(SRC_DIR.glob('*/*.ppm'))
    print(f"发现 {len(img_paths)} 张图片，开始生成雾霾数据...")

    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        if img is None: continue

        # 添加雾
        fog_img = add_fog(img, fog_intensity=0.8) 

        relative_path = img_path.relative_to(SRC_DIR)
        save_path = DST_DIR / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), fog_img)

    print(f"处理完成！雾霾数据集保存在: {DST_DIR}")

if __name__ == '__main__':
    process()