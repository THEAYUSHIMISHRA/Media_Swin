# 🧠 Medi_Swin  
### Degradation-Aware Medical Image Restoration using Swin Transformer + GAN

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![GPU](https://img.shields.io/badge/CUDA-Supported-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

**Medi_Swin** is a **hybrid Swin Transformer + GAN-based framework** designed to restore degraded chest X-ray images while preserving clinically critical details.

Unlike traditional CNN models, this system:

- Learns **global + local dependencies** using Swin Transformer  
- Uses **GAN training** for perceptual realism  
- Applies **dynamic degradation simulation** for real-world robustness  

🎯 **Goal:** Restore X-rays **without hallucination artifacts**, ensuring diagnostic reliability.

---

## 🚀 Key Highlights

- 🧠 **Swin Transformer Backbone (timm)**
- 🧩 **U-Net Style Encoder-Decoder with Skip Connections**
- ⚙️ **Degradation-Aware Training Pipeline**
- 🎭 **GAN Framework (Generator + PatchGAN Discriminator)**
- 🔬 **Medical Texture Preservation via Perceptual Loss**
- 🔄 **End-to-End Restoration Pipeline**

---

## 🏗️ Architecture

![Architecture](images/architecture.png)

### 🔹 Generator: MediSwin

- Backbone: `swin_tiny_patch4_window7_224` (pretrained)
- Decoder: Multi-stage upsampling with skip connections
- Output: Restored high-quality X-ray

✔ Combines **Transformer + CNN decoding power**

---

### 🔹 Discriminator: PatchGAN

- Uses **Spectral Normalization**
- Operates on image patches for fine realism
- Stabilizes GAN training

---

## ⚙️ Degradation Pipeline

![Degradation](images/degradation.png)

### 🔥 Training-Time Degradation (Dynamic)

- Gaussian Blur (80% probability)
- Noise Injection (σ ≈ 15)
- Downscaling (0.4x – 0.6x)

➡️ Enables robustness to real-world corrupted scans

---

## 📂 Dataset

- **Dataset:** COVID-19 Radiography Database  
- **Type:** Grayscale Chest X-rays  
- **Structure:**


---

## 🧪 Training Strategy

### 🔹 Two-Phase Training

#### 🟢 Phase 1: Baseline Learning (Epochs 1–25)
- Moderate degradation
- Focus: Texture reconstruction
- Optimizes PSNR & SSIM

#### 🔴 Phase 2: Stress Testing (Epochs 26–30)
- Heavy noise + aggressive scaling
- Focus: Real-world robustness

---

## ⏱️ Training Configuration

| Parameter | Value |
|----------|------|
| Epochs | 30 |
| Batch Size | 4 |
| Image Size | 224 |
| LR (Generator) | 5e-5 |
| LR (Discriminator) | 1e-5 |

---

## 💻 System Used

- CPU: Intel Xeon W-2175  
- RAM: 64 GB  
- GPU: NVIDIA Quadro P1000 (4GB)  
- Storage: 1.8 TB  

⚠️ Optimized for **CUDA GPUs (T4 / A100 recommended)**

---

## 📉 Loss Functions

### 🔹 Hybrid Loss Design

- **L1 Loss (×50)** → Pixel accuracy  
- **Perceptual Loss (×5)** → Texture realism (VGG16)  
- **GAN Loss (BCEWithLogits + Soft Labels)** → Adversarial training  

✔ Ensures both **numerical + visual fidelity**

---

## 📊 Results

| Mode | PSNR | SSIM |
|-----|------|------|
| 🟢 Baseline Performance| 44.54 dB | 0.9934 |
| 🔴 Stress-Test Performance | 30.75 dB | 0.9038 |

---


## 🖼️ Visual Results

![Results](images/results.png)

### ✨ Observations

- Removes **heavy noise & blur**
- Preserves **fine anatomical details**
- No hallucinated artifacts
- Maintains diagnostic integrity

---

## 📁 Project Structure
Medi_Swin/
│── arch/
│ ├── generator.py
│ ├── discriminator.py
│
│── data/
│ ├── raw/
│
│── dataset/
│ └── xray_dataset.py
│
│── utils/
│ ├── degradation.py
│ ├── losses.py
│ ├── metrics.py
│── config.py
│── train.py
│── test.py
│── unseen.py
│── visualize.py
│── requirements.txt

---
---

## 📦 Requirements

```txt id="requirements-block"
torch>=2.0
torchvision>=0.15
timm>=0.9
opencv-python
numpy
matplotlib
pillow
pandas
scikit-image
```

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/Medi_Swin.git
cd Medi_Swin

# Create environment
conda create -n mediswin python=3.10
conda activate mediswin

pip install -r requirements.txt
```
