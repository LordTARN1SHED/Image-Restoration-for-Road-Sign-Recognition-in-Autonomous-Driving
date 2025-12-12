# 13_pipeline_stress_test_multi.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from pathlib import Path
import os

# ================= 配置区域 =================
NUM_SAMPLES = 10  # 测试几张图？
CLEAN_DIR = Path('./data/gtsrb/GTSRB/Training')
OUTPUT_DIR = Path('./pipeline_results') # 结果保存文件夹

# 模型路径
MODEL_PATHS = {
    'Noise': './restoration_noise.pth',
    'Blur':  './restoration_blur.pth',
    'Fog':   './restoration_fog.pth'
}

# 畸变叠加顺序: Blur -> Fog -> Noise
# 修复执行顺序: Noise -> Fog -> Blur
RESTORATION_ORDER = ['Noise', 'Fog', 'Blur']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

# --- 1. 定义畸变函数 ---
def add_noise(image):
    img = image / 255.0
    noise = np.random.normal(0, 0.01 ** 0.5, img.shape) 
    out = img + noise
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)

def add_blur(image):
    degree = 5 
    angle = 45
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    return blurred

def add_fog(image):
    fog_intensity = 0.1 
    img = image / 255.0
    A = 0.9
    t = 1.0 - fog_intensity
    fog_img = img * t + A * (1 - t)
    return np.clip(fog_img * 255, 0, 255).astype(np.uint8)

# --- 2. 定义模型结构 ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.final = nn.Conv2d(64, 3, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        d2 = self.up2(b)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out

def get_vgg_prediction(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

def main():
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. 准备模型
    print("正在加载模型...")
    restoration_models = {}
    for name in MODEL_PATHS:
        if Path(MODEL_PATHS[name]).exists():
            net = SimpleUNet().to(DEVICE)
            net.load_state_dict(torch.load(MODEL_PATHS[name], map_location=DEVICE))
            net.eval()
            restoration_models[name] = net
        else:
            print(f"警告: 找不到模型 {MODEL_PATHS[name]}")
    
    vgg = models.vgg16(weights='DEFAULT')
    num_ftrs = vgg.classifier[6].in_features
    vgg.classifier[6] = nn.Linear(num_ftrs, 43)
    if Path('./vgg16_baseline.pth').exists():
        vgg.load_state_dict(torch.load('./vgg16_baseline.pth', map_location=DEVICE))
    vgg = vgg.to(DEVICE)
    vgg.eval()

    preprocess_vgg = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    preprocess_unet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 2. 随机采样图片
    all_files = list(CLEAN_DIR.glob('*/*.ppm'))
    if len(all_files) < NUM_SAMPLES:
        selected_files = all_files
    else:
        selected_files = random.sample(all_files, NUM_SAMPLES)
    
    print(f"已选中 {len(selected_files)} 张图片进行测试...\n")

    # 统计数据容器
    stats_clean_conf = []
    stats_bad_conf = []
    stats_restored_conf = []

    # 3. 循环处理每一张图
    for idx, img_path in enumerate(selected_files):
        print(f"[{idx+1}/{NUM_SAMPLES}] 处理: {img_path.name}")
        
        # 读取原图
        original_cv = cv2.imread(str(img_path))
        original_cv = cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB)
        
        history_images = [original_cv]
        history_titles = ["Original"]
        
        # --- Phase 1: 叠加畸变 (Blur -> Fog -> Noise) ---
        current_img = original_cv.copy()
        
        # Blur
        current_img = add_blur(current_img)
        history_images.append(current_img)
        history_titles.append("+ Blur")
        
        # Fog
        current_img = add_fog(current_img)
        history_images.append(current_img)
        history_titles.append("+ Fog")
        
        # Noise
        current_img = add_noise(current_img)
        history_images.append(current_img)
        history_titles.append("+ Noise (Input)")
        
        # 备份坏图用于记录 VGG 分数
        bad_img_for_stats = current_img.copy()

        # --- Phase 2: 级联修复 (Noise -> Fog -> Blur) ---
        tensor_img = preprocess_unet(Image.fromarray(current_img)).unsqueeze(0).to(DEVICE)
        
        for model_name in RESTORATION_ORDER:
            if model_name in restoration_models:
                net = restoration_models[model_name]
                with torch.no_grad():
                    tensor_img = net(tensor_img)
                    
                    # 可视化保存
                    vis_tensor = torch.clamp(tensor_img, 0, 1)
                    vis_img = vis_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
                    vis_img = (vis_img * 255).astype(np.uint8)
                    
                    history_images.append(vis_img)
                    history_titles.append(f"After {model_name}")

        # --- Phase 3: VGG 评分与保存图片 ---
        plt.figure(figsize=(20, 8))
        
        # 临时变量记录这张图的三个关键分数
        conf_c, conf_b, conf_r = 0, 0, 0

        for i, (img, title) in enumerate(zip(history_images, history_titles)):
            pil_img = Image.fromarray(img)
            vgg_input = preprocess_vgg(pil_img).unsqueeze(0).to(DEVICE)
            
            pred_cls, conf = get_vgg_prediction(vgg, vgg_input)
            
            # 记录关键节点的置信度
            if i == 0: conf_c = conf          # Original
            if i == 3: conf_b = conf          # + Noise (Final Bad)
            if i == 6: conf_r = conf          # Final Restored
            
            ax = plt.subplot(2, 4, i + 1)
            ax.imshow(img)
            color = 'green' if conf > 0.8 else ('orange' if conf > 0.5 else 'red')
            ax.set_title(f"{title}\nPred: {pred_cls} | Conf: {conf:.2f}", color=color, fontsize=12, fontweight='bold')
            ax.axis('off')

        # 保存图片
        save_path = OUTPUT_DIR / f'pipeline_sample_{idx+1}.png'
        plt.tight_layout()
        plt.savefig(str(save_path))
        plt.close() # 关闭画板释放内存
        
        # 添加到统计
        stats_clean_conf.append(conf_c)
        stats_bad_conf.append(conf_b)
        stats_restored_conf.append(conf_r)

    # 4. 打印最终统计报告
    avg_clean = sum(stats_clean_conf) / len(stats_clean_conf)
    avg_bad = sum(stats_bad_conf) / len(stats_bad_conf)
    avg_restored = sum(stats_restored_conf) / len(stats_restored_conf)

    print("\n" + "="*40)
    print(f"最终测试报告 (共 {NUM_SAMPLES} 张图片)")
    print("="*40)
    print(f"{'Stage':<20} | {'Avg Confidence':<15}")
    print("-" * 38)
    print(f"{'Original (Clean)':<20} | {avg_clean:.4f}")
    print(f"{'Distorted (Input)':<20} | {avg_bad:.4f}")
    print(f"{'Restored (Output)':<20} | {avg_restored:.4f}")
    print("="*40)
    print(f"所有结果图已保存在: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()