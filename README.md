 **Disclaimer:** This README was generated with the assistance of a Large Language Model (LLM), based on the author's original project report and code.

# Texture Anomaly Detection: Self-Supervised MoCo vs. Low-Level Processing 

This repository contains the codebase for the project developed for the **Signal, Image and Video** course (Master's Degree in Artificial Intelligence Systems) [1]. 

The goal of this project is to tackle the industrial problem of texture anomaly (defect) detection without relying on explicitly annotated defective samples [2]. To achieve this, the project compares a traditional middle-level statistical pipeline with a modern self-supervised deep learning architecture based on Momentum Contrast (MoCo) [2, 3].

## 📊 Dataset and Synthetic Anomalies
The project utilizes the **Describable Textures Dataset (DTD)**, with images converted to grayscale and resized to 224x224 pixels [3]. 
Since the dataset lacks anomalous samples, synthetic defects were deliberately injected (e.g., impulsive Salt & Pepper noise, localized structural alterations) to simulate real-world surface damages like scratches and stains [4].

## ⚙️ Implemented Pipelines

### 1. Classic Statistical Baseline (Middle-Level Processing)
A traditional computer vision approach that requires no training [5]:
* **Feature Extraction:** Uses a uniform Local Binary Pattern (LBP) operator (Radius=3, 24 sample points) to encode spatial information [6].
* **Disorder Measurement:** Computes local entropy using a circular kernel (Radius=15) to detect disruptions in the normal texture patterns [6].
* **Segmentation:** Applies K-Means clustering (K=2) to isolate anomalous pixels from the healthy background [7].

### 2. Self-Supervised Learning (MoCo)
A deep learning approach designed to learn robust representations without labels [8]:
* **Architecture:** Utilizes a custom ResNet18-based encoder (modified for single-channel grayscale input) and a 128-dimensional Projection Head MLP [8, 9].
* **Custom Low-Level Data Augmentation:** Instead of using black-box libraries, the data augmentation engine necessary to create positive pairs for contrastive learning was programmed from scratch [10]. It includes spatial filtering (Median filters, 2D discrete convolutions with dynamically generated Gaussian kernels) and statistical manipulations (Histogram Equalization via Cumulative Distribution Function) [10].
* **Training:** Optimized using a dynamic dictionary (FIFO queue of size 1024) over a highly restricted regimen of only 15 epochs [11, 12].

## 📈 Results
The performance of both pipelines was evaluated using the **Area Under the Receiver Operating Characteristic Curve (AUC ROC)**, chosen for its robustness against significant class imbalances [13].

| Method | AUC ROC |
| :--- | :--- |
| Classic Baseline (LBP + Entropy + K-Means) | 0.5055 |
| **Self-Supervised MoCo (15 epochs)** | **0.6653** |

Despite the extremely limited training phase, the MoCo framework demonstrated superior adaptability and robustness in identifying complex structural deviations compared to the traditional baseline [14, 15].
