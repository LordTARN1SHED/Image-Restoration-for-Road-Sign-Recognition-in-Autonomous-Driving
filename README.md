# Project Report: Image Restoration for Road Sign Recognition in Autonomous Driving

**Course:** ECE 253 Fall 2025 (UCSD)
**Team Members:** Bingrui Zhang, Jialin Shang, Zhexi Feng
**Date:** November 23, 2025

---

## 1. Executive Summary
This project addresses a critical safety challenge in autonomous driving: **Traffic Sign Recognition (TSR) under adverse weather conditions**. While modern Convolutional Neural Networks (CNNs) like VGG16 achieve near-perfect accuracy on clear images, their performance degrades catastrophically in the presence of environmental distortions such as rain noise, motion blur, and heavy fog.

We propose and implement a **"Restoration-First" pipeline**. Instead of retraining the classification model on distorted data, we employ deep learning-based restoration models to recover image quality before feeding it into the classifier. We explored two strategies:
1.  **Specialized Restoration:** Dedicated U-Nets for specific distortions (Noise, Blur, Fog).
2.  **Unified Blind Restoration:** A single ResUNet capable of handling unknown, compound distortions (e.g., Fog + Blur + Noise simultaneously).

**Key Achievements:**
* **Noise Recovery:** Accuracy improved from **26.82%** $\to$ **75.99%**.
* **Fog Recovery:** Accuracy improved from **61.07%** $\to$ **90.17%**.
* **Blur Recovery:** Accuracy improved from **55.57%** $\to$ **71.37%** (via Perceptual Loss).
* **Compound Distortion:** Accuracy salvaged from **5.78%** $\to$ **33.03%** (a 6x relative improvement under extreme conditions).

---

## 2. Project Macro Logic & Architecture

The project is structured around a modular pipeline consisting of four distinct stages:

### Stage 1: Simulation (The Problem)
We simulate real-world degradations on the GTSRB (German Traffic Sign Recognition Benchmark) dataset.
* **Single Distortion:** Additive White Gaussian Noise, Linear Motion Blur, and Atmospheric Haze.
* **Compound Distortion:** A "Nightmare Scenario" stacking **Blur + Fog + Noise** sequentially to test extreme robustness.

### Stage 2: Restoration (The Solution)
We trained two types of restoration models:
* **Standard U-Net (Pixel-wise):** Effective for stochastic noise and global haze removal.
* **Advanced ResUNet (Perceptual):** Uses **VGG Feature Loss** to recover sharp edges in blurred images and handle complex compound distortions.

### Stage 3: Recognition (The Validation)
We use a pre-trained **VGG16 classifier** as a fixed "Judge." The success of restoration is measured by whether the VGG16 model can correctly classify the restored image, linking visual quality to semantic interpretability.

### Stage 4: Interpretability (The "Why")
We employ **Hidden State Visualization** and **UMAP Dimensionality Reduction** to peer inside the VGG16 "black box," proving that our restoration models recover the actual feature distributions required for classification.

---

## 3. File Structure & Implementation Details

Below is a comprehensive guide to the codebase.

### A. Data Preparation Phase
* **`01_download_data.py`**: Automatically downloads GTSRB using `torchvision`.
* **`02_gen_noise.py`**: Generates the "Noisy" dataset using AWGN ($\sigma \approx 0.02$).
* **`03_gen_blur.py`**: Generates the "Blurred" dataset using linear motion kernels.
* **`04_gen_fog.py`**: Generates the "Foggy" dataset using the Atmospheric Scattering Model ($I = Jt + A(1-t)$).
* **`16_gen_compound_data.py`**: Generates the "Compound" dataset (Blur $\to$ Fog $\to$ Noise) for the final stress test.

### B. Baseline & Validation Phase
* **`05_train_baseline.py`**: Fine-tunes a VGG16 on clean data (Accuracy: **99.96%**).
* **`06_test_baseline.py`**: The universal testing harness for single distortions.

### C. Specialized Restoration (Single Distortion)
* **`07_train_restoration.py`**: Trains standard U-Net with MSE Loss. Effective for **Noise** and **Fog**.
* **`07_train_restoration_advanced.py`**: Trains U-Net with **Perceptual Loss (VGG Features)** + L1 Loss. Specifically designed for **Motion Blur** to prevent over-smoothing.
* **`08_run_inference.py`**: Batch processes the dataset using the trained specialized models.

### D. Advanced Analysis (Interpretability)
* **`11_visualize_hidden_states_v2.py`**: 
    * Extracts intermediate feature maps (Layer 2) and final convolutional outputs (Layer 30) from VGG16.
    * **Goal:** Visual proof that restoration recovers edge features (activations) that were lost in distorted images.
* **`12_generate_umap_pt.py`**: 
    * Performs Global Average Pooling (GAP) on VGG features for 700+ images.
    * Uses **UMAP** to project high-dimensional features (512-d) into 2D space.
    * **Goal:** Shows that restored images "migrate" back to the cluster of clean images in the feature space.

### E. Unified Restoration (Compound Distortion)
* **`14_train_unified_advanced.py`**: 
    * **Model:** A deeper **ResUNet** (U-Net with Residual Blocks).
    * **Training:** Uses **Dynamic Distortion Generation**, creating random mixtures of Noise, Blur, and Fog on-the-fly during training.
    * **Loss:** Perceptual Loss + L1 Loss.
* **`17_run_unified_inference.py`**: Batch processes the "Compound" dataset using the Unified ResUNet.
* **`18_test_unified_benchmark.py`**: Evaluates the Unified Model on the compound dataset.

---

## 4. Final Experimental Results

### Part 1: Single Distortion Results (Specialized Models)
The table below summarizes the Top-1 Accuracy of VGG16 on single distortions.

| Dataset Condition | Accuracy | Improvement | Analysis |
| :--- | :--- | :--- | :--- |
| **Clean (Baseline)** | **99.96%** | - | Theoretical upper bound. |
| | | | |
| **Noisy (Bad)** | 26.82% | - | Features destroyed by high-frequency variance. |
| **Restored (Noise)** | **75.99%** | **+49.17%** | **Major Success.** U-Net effectively learned denoising. |
| | | | |
| **Foggy (Bad)** | 61.07% | - | Low contrast hampers edge detection. |
| **Restored (Fog)** | **90.17%** | **+29.10%** | **Near Perfect.** Dehazing recovered semantic contrast. |
| | | | |
| **Blurred (Bad)** | 55.57% | - | Edges destroyed by motion integration. |
| **Restored (Blur)** | **71.37%** | **+15.80%** | **Hard-won Success.** (MSE Loss only achieved 51%; Perceptual Loss was critical here). |

### Part 2: Compound Distortion Results (Unified Model)
This test simulates a "Nightmare Scenario" where images are hit by Blur, Fog, and Noise simultaneously.

| Dataset Condition | Accuracy | Improvement | Analysis |
| :--- | :--- | :--- | :--- |
| **Clean (Baseline)** | 99.96% | - | - |
| **Compound Distorted** | **5.78%** | - | **System Failure.** The model is essentially blind. |
| **Unified Restored** | **33.03%** | **~6x Rel.** | **Recovery.** While not perfect, the model salvages usable features from near-zero information. |

---

## 5. Discussion & Key Findings

### 1. Perceptual Loss is Critical for De-blurring
Our initial experiments with MSE Loss for de-blurring failed (Accuracy dropped from 55% $\to$ 51%). MSE favors "safe" average pixel values, resulting in waxy, over-smoothed images that lack the sharp edges VGG16 needs. Switching to **Perceptual Loss** forced the U-Net to reconstruct high-frequency edge information, boosting accuracy to **71.37%**.

### 2. Feature Space Alignment (UMAP Analysis)
Our UMAP visualization confirmed that:
* **Distorted images** (Red points) are scattered far from the clean cluster, indicating a "covariate shift."
* **Restored images** (Blue points) successfully migrate back towards the **Clean images** (Green cluster).
This proves that our restoration models are not just making images "look nice" to humans, but are mathematically aligning them with the distribution that the classifier expects.

### 3. The Limits of Information Theory (Compound Distortion)
In the compound test, the input accuracy was **5.78%**, meaning the signal-to-noise ratio was extremely low. Recovering to **33.03%** represents a massive relative improvement (**~600%**). The model cannot invent information that is completely lost, but it successfully acts as a "Rescue Filter," salvaging whatever semantic structures remain to make the system significantly safer than a blind guess.

---
## 6. Usage Guide & Quick Start

The project scripts are numbered and designed to be run sequentially.

**⚠️ IMPORTANT NOTE: Missing Judge Model**
Due to file size limitations, the pre-trained VGG16 Judge Model (`vgg16_baseline.pth`) is **NOT** included in the repository. However, we have provided the 4 pre-trained restoration models (`restoration_noise.pth`, `restoration_blur.pth`, `restoration_fog.pth`, `restoration_unified_resnet.pth`).

**If you want to test our results using the pre-trained restoration models, please follow this specific order:**

### Step 1: Preparation
1.  **Download Dataset:**
    Run `python 01_download_data.py` to download and extract the GTSRB dataset.
2.  **Generate the Judge (Required for Scoring):**
    Since the judge model is missing, you must generate it locally (this takes ~5-10 mins on a GPU).
    Run `python 05_train_baseline.py`.
    *This will save `vgg16_baseline.pth` locally, enabling all subsequent testing scripts.*

### Step 2: Generate Distorted Data
Run the following scripts to create the testing samples:
* `python 02_gen_noise.py`
* `python 03_gen_blur.py`
* `python 04_gen_fog.py`

### Step 3: Run Restoration (Using Our Pre-trained Models)
We have provided the trained `.pth` files. You can directly run inference without retraining the U-Nets.
* `python 08_run_inference.py`
    * *This uses our `restoration_*.pth` models to repair the images generated in Step 2 and saves them to `data/restored/`.*

### Step 4: Visualize & Test
* **Visual Comparison:**
    Run `python 10_visualize_result.py` to see a side-by-side comparison (Original vs. Bad vs. Restored).
* **Unified Model Stress Test:**
    Run `python 15_test_unified.py`
    * *This simulates the "Nightmare Scenario" (Blur+Fog+Noise), runs our Unified ResNet, and uses the VGG model (from Step 1) to display confidence scores.*

---

### Full Script List Reference
If you wish to retrain everything from scratch, simply run files `01` through `18` in numerical order.

* `01-04`: Data Setup
* `05-06`: Baseline Training & Testing
* `07`: Train Specialized U-Nets (Noise/Blur/Fog)
* `08-10`: Inference & Visualization
* `11-12`: Hidden State & UMAP Analysis
* `13`: Pipeline Stress Test
* `14`: Train Unified ResNet
* `15-18`: Unified Model Benchmarking
---
## 7. Conclusion

This project demonstrates that **Restoration-First** is a viable paradigm for robust autonomous driving perception.

1.  **Specialized models** can restore accuracy to near-baseline levels (up to 90%) for single weather events.
2.  **Advanced Loss functions** (Perceptual Loss) are necessary for structural distortions like blur.
3.  **Unified Blind Restoration** is possible using ResUNet and dynamic training, offering a safety net for extreme, multi-weather scenarios where standard models fail completely.

By peeling back the layers of distortion, we ensure that the "eyes" of the autonomous vehicle remain open, even when the world outside is dark and stormy.