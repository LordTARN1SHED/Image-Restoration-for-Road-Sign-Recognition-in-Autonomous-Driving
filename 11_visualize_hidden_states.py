# 11_visualize_hidden_states.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

# ================= 配置 =================
# 选一张比较典型的图（比如限速标志，边缘清晰）
# 请替换成你本地存在的某张图片的相对路径
SAMPLE_IMG_REL_PATH = '00001/00025_00029.ppm' 

# 基础路径
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
    提取 VGG16 指定层的特征图
    layer_index=0 -> 第1个卷积层 (Conv1_1)
    layer_index=2 -> 第2个卷积层 (Conv1_2)
    layer_index=5 -> 第3个卷积层 (Conv2_1) ... 以此类推
    """
    # 截取 VGG 的前 N 层
    feature_extractor = model.features[:layer_index+1]
    
    with torch.no_grad():
        features = feature_extractor(img_tensor)
    
    return features

def plot_heatmap(features):
    """
    将多通道特征图平均，转为单通道热力图用于可视化
    """
    # features shape: [1, 64, 224, 224]
    # 对 Channel 维度求平均 -> [1, 224, 224]
    heatmap = torch.mean(features, dim=1).squeeze().cpu().numpy()
    
    # 归一化到 0-1 之间以便绘图
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap

def main():
    # 1. 准备模型 (不需要训练好的权重，只需要 ImageNet 预训练的提取特征能力即可)
    # 因为我们想看的是 VGG "通用" 的视觉能力如何受影响
    print("加载 VGG16 模型...")
    vgg = models.vgg16(weights='DEFAULT').to(DEVICE)
    vgg.eval()
    
    # 2. 准备预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 这里不Normalize也是可以的，为了看原始激活更直观，或者保持一致
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 寻找文件
    clean_path = CLEAN_DIR / SAMPLE_IMG_REL_PATH
    if not clean_path.exists():
        print(f"错误: 找不到文件 {clean_path}，请修改脚本里的 SAMPLE_IMG_REL_PATH")
        return

    # 4. 开始绘图
    # 我们对比三种畸变，每种一行
    # 每行显示: Clean Feature | Bad Feature | Restored Feature
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    # 列标题
    cols = ["Input Image (Bad/Restored)", "Clean Features", "Distorted Features", "Restored Features"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold')

    tasks = ['Noise', 'Blur', 'Fog']
    
    # 选择查看 VGG 的第几层？
    # 第 2 层 (ReLU后) 通常能看到很清晰的边缘
    TARGET_LAYER = 2 

    print(f"正在提取 VGG16 第 {TARGET_LAYER} 层的 Hidden States...")

    for i, task in enumerate(tasks):
        # 构建路径
        bad_path = BAD_DIRS[task] / SAMPLE_IMG_REL_PATH
        if not bad_path.exists(): bad_path = bad_path.with_suffix('.png')
        
        restored_path = RESTORED_DIRS[task] / SAMPLE_IMG_REL_PATH
        restored_path = restored_path.with_suffix('.png')

        # 读取图片
        img_clean = Image.open(clean_path).convert('RGB')
        img_bad = Image.open(bad_path).convert('RGB')
        img_res = Image.open(restored_path).convert('RGB')

        # 转 Tensor
        t_clean = transform(img_clean).unsqueeze(0).to(DEVICE)
        t_bad = transform(img_bad).unsqueeze(0).to(DEVICE)
        t_res = transform(img_res).unsqueeze(0).to(DEVICE)

        # 提取特征
        f_clean = get_vgg_feature_maps(vgg, t_clean, TARGET_LAYER)
        f_bad = get_vgg_feature_maps(vgg, t_bad, TARGET_LAYER)
        f_res = get_vgg_feature_maps(vgg, t_res, TARGET_LAYER)

        # 转热力图
        h_clean = plot_heatmap(f_clean)
        h_bad = plot_heatmap(f_bad)
        h_res = plot_heatmap(f_res)

        # --- 绘图 ---
        # 第一列：展示修好的那张图 (RGB)
        axes[i, 0].imshow(img_res)
        axes[i, 0].set_ylabel(task, fontsize=14, fontweight='bold')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # 第二列：Clean 特征
        axes[i, 1].imshow(h_clean, cmap='viridis')
        axes[i, 1].axis('off')

        # 第三列：坏图 特征
        axes[i, 2].imshow(h_bad, cmap='viridis')
        axes[i, 2].axis('off')

        # 第四列：修好图 特征
        axes[i, 3].imshow(h_res, cmap='viridis')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('hidden_state_visualization.png')
    print("可视化完成！已保存为 hidden_state_visualization.png")
    plt.show()

if __name__ == '__main__':
    main()