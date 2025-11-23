# Image Restoration for Road Sign Recognition in Autonomous Driving

**Course:** ECE 253 Fall 2025 (UCSD)  
**Team Members:** Bingrui Zhang, Jialin Shang, Zhexi Feng

---

## 1. Project Overview

### Motivation
In autonomous driving systems, robust traffic sign recognition is critical for safety. However, real-world conditions often degrade image quality due to adverse weather (rain, fog) or camera instability (motion blur). Standard recognition models trained on high-quality data often fail catastrophically under such distortions.

### Core Hypothesis
A **“Restoration → Recognition”** pipeline—where images are first cleaned by a restoration network before classification—can provide a more robust solution than naive data augmentation, especially under extreme or variable environmental conditions.

### Technical Approach
1. **Stage 1 (Restoration):** Implement a U-Net–based autoencoder to restore degraded images (Noise, Blur, Fog) back to clean quality.  
2. **Stage 2 (Recognition):** Feed restored images into a pre-trained VGG16 classifier to evaluate recovery in recognition accuracy.

---

## 2. Implementation Workflow

Our engineering workflow follows a modular structure, from data preparation to restoration model training.

### File Structure

Project_Root/
├── data/
│ ├── gtsrb/ # Raw GTSRB dataset
│ └── processed/ # Distorted datasets (Noise, Blur, Fog)
├── 01_download_data.py # Data acquisition
├── 02_gen_noise.py # Gaussian noise simulation
├── 03_gen_blur.py # Motion blur simulation
├── 04_gen_fog.py # Fog/haze synthesis
├── 05_train_baseline.py # VGG16 training (clean data)
├── 06_test_baseline.py # Baseline testing
├── 07_train_restoration.py # U-Net restoration model training
├── vgg16_baseline.pth # Saved classifier weights
└── restoration_*.pth # Saved restoration weights

---

## 3. Detailed Implementation & Current Progress

### 3.1 Data Acquisition (`01_download_data.py`)
- **Dataset:** GTSRB (German Traffic Sign Recognition Benchmark)  
- **Implementation:** Using `torchvision.datasets.GTSRB`  
- **Classes:** 43 traffic sign categories  

---

### 3.2 Distortion Simulation

We generated three synthetic degraded datasets to mimic real-world image corruption.

#### A. Noise Simulation (`02_gen_noise.py`)
- **Method:** Additive White Gaussian Noise (AWGN)  
- **Variance:** `var = 0.02`  
- **Effect:** Simulates low-light or sensor noise  
- **CNN Impact:** Destroys texture; causes severe accuracy drop  

#### B. Motion Blur Simulation (`03_gen_blur.py`)
- **Method:** Convolution with linear motion kernel  
- **Parameters:** `degree = 12`, `angle = 45°`  
- **Effect:** Simulates camera shake / high-speed motion  
- **Impact:** Edge information lost; recognition impaired  

#### C. Fog Simulation (`04_gen_fog.py`)
- **Method:** Atmospheric scattering model  
- **Equation:**  
  \[
  I(x) = J(x)t(x) + A(1 - t(x))
  \]
- **Fog Intensity:** Random range **0.85–0.95**  
- **Impact:** Reduces contrast; produces realistic high-opacity haze  

---

### 3.3 Baseline Recognition Model (`05_train_baseline.py`)
- **Model:** VGG16 (pre-trained on ImageNet)  
- **Modification:** Final FC layer replaced with 43-class output  
- **Training Data:** Clean GTSRB dataset  
- **Result:** **99.96% accuracy** on clean validation set  

---

### 3.4 Baseline Benchmarking Results (`06_test_baseline.py`)

| Dataset | Accuracy | Interpretation |
|--------|----------|----------------|
| **Clean** | **99.96%** | Upper bound; model nearly perfect |
| **Noise** | **26.82%** | Critical failure; unusable |
| **Blur** | **55.57%** | Major degradation |
| **Fog** | **61.07%** | Substantial confusion |

**Conclusion:** Distortions successfully create challenging test scenarios, validating the need for restoration networks.

---

### 3.5 Restoration Network (`07_train_restoration.py`)

- **Architecture:** U-Net (Autoencoder with skip connections)  
  - **Encoder:** Downsampling for global context  
  - **Decoder:** Upsampling for reconstruction  
  - **Skip Connections:** Preserve spatial details  
- **Loss:** Mean Squared Error (MSE)  
- **Status:** Three models in training:
  - Noise → Clean  
  - Blur → Clean  
  - Fog → Clean  

---

## 4. Next Steps

1. **Finalize Restoration Training**  
   Continue training U-Net models for all three distortion types.

2. **Generate Restored Dataset**  
   Feed distorted images through trained U-Net models.

3. **Re-Evaluate Recognition**  
   Run the fixed VGG16 baseline:
   - Expected improvements (e.g., Noise 26% → ~80%).

4. **Collect Real-World San Diego Data**  
   Evaluate generalization on real fog/rain/low-light scenes.

---
