# Project Report: Image Restoration for Road Sign Recognition in Autonomous Driving

**Team Members:** BZ, JS, ZF
**Date:** November 23, 2025

---

## 1. Executive Summary
This project addresses a critical safety challenge in autonomous driving: **Traffic Sign Recognition (TSR) under adverse weather conditions**. While modern Convolutional Neural Networks (CNNs) like VGG16 achieve near-perfect accuracy on clear images, their performance degrades catastrophically in the presence of environmental distortions such as rain noise, motion blur, and heavy fog.

We propose and implement a **"Restoration-First" pipeline**. Instead of retraining the classification model on distorted data, we employ a U-Net-based Autoencoder to restore the image quality before feeding it into the classifier. 

**Key Achievement:**
* **Noise:** Accuracy recovered from **26.82%** to **75.99%**.
* **Fog:** Accuracy recovered from **61.07%** to **90.17%**.
* **Blur:** Accuracy recovered from **55.57%** to **71.37%** (using Advanced Perceptual Loss).

---

## 2. Project Macro Logic & Architecture

The project is structured around a modular pipeline consisting of three distinct stages:

### Stage 1: Simulation (The Problem)
We simulate real-world degradations on the GTSRB (German Traffic Sign Recognition Benchmark) dataset. This establishes a controlled environment to quantify how much "weather" breaks a standard AI model.

### Stage 2: Restoration (The Solution)
We train deep learning models (Autoencoders/U-Nets) to map distorted images back to their clean counterparts.
* **Standard Approach:** Uses Mean Squared Error (MSE) Loss for pixel-wise reconstruction (effective for Noise and Fog).
* **Advanced Approach:** Uses **Perceptual Loss (VGG Feature Loss)** for Motion Blur to prevent over-smoothing and recover sharp edges required for classification.

### Stage 3: Recognition (The Validation)
We use a pre-trained VGG16 classifier as a "Judge." The success of the restoration is measured not just by how good the image looks to the human eye (PSNR), but by whether the VGG16 model can correctly classify the sign after restoration.

---

## 3. File Structure & Implementation Details

Below is a detailed explanation of every script in our repository and its role in the pipeline.

### A. Data Preparation Phase

* **`01_download_data.py`**
    * **Function:** Automatically downloads the GTSRB dataset using `torchvision`.
    * **Logic:** Sets up the raw directory structure required for training.

* **`02_gen_noise.py`** (Distortion: Noise)
    * **Function:** Generates the "Noisy" dataset.
    * **Logic:** Adds Additive White Gaussian Noise (AWGN) to simulate sensor noise or heavy rain grain. The variance is tuned to degrade VGG16 accuracy significantly (to ~26%).

* **`03_gen_blur.py`** (Distortion: Motion Blur)
    * **Function:** Generates the "Blurred" dataset.
    * **Logic:** Creates a linear motion kernel (mimicking a camera shutter open while the car is moving) and convolves it with the clean image. This destroys high-frequency edge information.

* **`04_gen_fog.py`** (Distortion: Haze/Fog)
    * **Function:** Generates the "Foggy" dataset.
    * **Logic:** Implements the Atmospheric Scattering Model: $I(x) = J(x)t(x) + A(1-t(x))$. We use a randomized high fog intensity ($0.85 - 0.95$) to ensure the distortion is severe enough to challenge the model (dropping accuracy to ~61%).

### B. Baseline & Validation Phase

* **`05_train_baseline.py`**
    * **Function:** Trains the Classification Model.
    * **Implementation:** Uses a **VGG16** architecture pre-trained on ImageNet. We modify the final fully connected layer to output 43 classes (for GTSRB) and fine-tune it on **Clean Data only**.
    * **Result:** Achieves **99.96%** accuracy on the validation set, serving as the "Gold Standard."

* **`06_test_baseline.py`** (and `09_test_baseline.py`)
    * **Function:** The universal testing harness.
    * **Logic:** Loads the trained VGG16 model and evaluates it across all datasets (Clean, Noisy, Blurred, Foggy, and the Restored versions). This script produces the final "Scorecard" for the project.

### C. Restoration Phase (The Core)

* **`07_train_restoration.py`** (Standard U-Net)
    * **Function:** Trains the restoration model for **Noise** and **Fog**.
    * **Architecture:** A symmetric U-Net with Skip Connections.
    * **Loss Function:** **MSE Loss (L2)**.
    * **Logic:** MSE effectively averages out random noise and learns to subtract the global white haze. It works perfectly for these two domains.

* **`07_train_restoration_advanced.py`** (Advanced U-Net)
    * **Function:** Trains the restoration model specifically for **Motion Blur**.
    * **Why a separate script?** Our experiments showed that MSE Loss caused "over-smoothing" on blurred images, actually lowering accuracy (to ~51%).
    * **Innovation:** We introduced **Perceptual Loss**. We feed both the generated image and real image into a frozen VGG network and minimize the difference in their *feature maps* (Perception) rather than just pixels. This forces the U-Net to reconstruct sharp edges.

* **`08_run_inference.py`**
    * **Function:** Pipeline Integration.
    * **Logic:** It loads the trained `.pth` restoration models, processes the entire distorted dataset, and saves the repaired images to `data/restored/`. It also calculates image quality metrics (**PSNR** and **SSIM**) to objectively measure visual improvement.

### D. Visualization

* **`10_visualize_result.py`**
    * **Function:** Qualitative Analysis.
    * **Logic:** Randomly selects samples and creates a side-by-side comparison grid (Original vs. Distorted vs. Restored) for visual inspection and report generation.

---

## 4. Final Experimental Results

The table below summarizes the Top-1 Classification Accuracy of the VGG16 model across different conditions.

| Dataset Condition | Accuracy | Improvement | Analysis |
| :--- | :--- | :--- | :--- |
| **Clean (Baseline)** | **99.96%** | - | The theoretical upper bound. |
| | | | |
| **Noisy (Bad)** | 26.82% | - | Feature extraction failed completely. |
| **Restored (Noise)** | **75.99%** | **+49.17%** | **Major Success.** The U-Net effectively denoised the signal. |
| | | | |
| **Foggy (Bad)** | 61.07% | - | Low contrast prevented edge detection. |
| **Restored (Fog)** | **90.17%** | **+29.10%** | **Near Perfect.** Dehazing recovered almost all semantic information. |
| | | | |
| **Blurred (Bad)** | 55.57% | - | Edges were destroyed by motion kernels. |
| **Restored (Blur)** | **71.37%** | **+15.80%** | **Hard-won Success.** *Note: Using standard MSE loss yielded only 51%. Switching to Perceptual Loss achieved 71%.* |

---

## 5. Discussion & Key Findings

### 1. The Effectiveness of MSE on Noise and Haze
For stochastic distortions like Gaussian Noise, the standard Mean Squared Error (MSE) loss function is highly effective. Since noise has a zero-mean distribution, the U-Net learns to output the average pixel value, effectively canceling out the grain. Similarly, fog acts as a global pixel intensity shift (whitening), which MSE can easily learn to reverse.

### 2. The Challenge of De-blurring (The "Negative Result" turned Success)
Initially, our de-blurring model using MSE Loss achieved only **51.54%** accuracy, which was *worse* than the distorted input (55.57%).
* **Reason:** MSE tends to generate "safe", smooth averages. For a blurred edge, the "average" is still a smooth gradient, not a sharp line. The VGG classifier relies heavily on sharp gradients (edges) to identify shapes.
* **Solution:** By implementing **Perceptual Loss** (via `07_train_restoration_advanced.py`), we optimized the model to match the *high-level features* of the clean image. This forced the generator to reconstruct sharp boundaries, successfully raising the accuracy to **71.37%**.

---

## 6. Conclusion

This project demonstrates that **image restoration is a viable and effective preprocessing step for autonomous driving perception systems**.

We successfully built a robust pipeline that:
1.  Simulates realistic driving hazards.
2.  Restores visual quality using deep learning.
3.  Significantly recovers the downstream classification accuracy of a black-box model.

The distinction between **Pixel-based Loss** (for noise/fog) and **Perceptual Loss** (for blur) proved to be a critical insight, highlighting that "visual similarity" (PSNR) does not always equal "semantic interpretability" (Accuracy) for AI models.