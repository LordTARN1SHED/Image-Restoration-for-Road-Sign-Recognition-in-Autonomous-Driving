# 02_gen_noise.py
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# 配置路径
SRC_DIR = Path('./data/gtsrb/GTSRB/Training')
DST_DIR = Path('./data/processed/Noise')

def add_gaussian_noise(image, mean=0, var=0.01):
    """
    添加高斯噪声
    image: 原始图片 (0-255)
    var: 方差，控制噪声强度
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
        print(f"错误：找不到源数据目录 {SRC_DIR}")
        return

    # 遍历所有类别文件夹
    img_paths = list(SRC_DIR.glob('*/*.ppm')) # GTSRB原格式是.ppm
    print(f"发现 {len(img_paths)} 张图片，开始生成噪点数据...")

    for img_path in tqdm(img_paths):
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None: continue

        # 添加噪声
        noisy_img = add_gaussian_noise(img, var=0.02) # 调整var可以改变噪声大小

        # 保持目录结构
        relative_path = img_path.relative_to(SRC_DIR)
        save_path = DST_DIR / relative_path
        
        # 创建对应的子文件夹
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存 (转为png通用格式，或者保持ppm)
        cv2.imwrite(str(save_path), noisy_img)

    print(f"处理完成！噪点数据集保存在: {DST_DIR}")

if __name__ == '__main__':
    process()