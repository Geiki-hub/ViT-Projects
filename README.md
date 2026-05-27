# ViT-Projects
This repository contains the official implementation and benchmarking suite for evaluating Vision Transformer (ViT) architectures to automate MYC status prediction from digital histopathology whole-slide images (WSIs) of Diffuse Large B-Cell Lymphoma (DLBCL).

This pipeline offers a high-throughput, mathematically validated alternative to resource-intensive Fluorescence In Situ Hybridization (FISH) screening by processing standard H&E stained slides.

# Efficiency and Interpretability in MYC Status Prediction: A Comparative Study of Vision Transformers
## Project Overview

Identifying MYC status is essential for clinical risk stratification in DLBCL. This project benchmarks five state-of-the-art Vision Transformer configurations across critical diagnostic axes: predictive metrics (Accuracy, AUROC), statistical significance (DeLong's test, 95% CIs), and computational throughput.

[Raw 120 WSIs] ➔ [Macenko Stain Normalization] ➔ [Lossless Compression] ➔ [463,200 Patches] ➔ [ViT Benchmarking]

## Key Features & Pipeline

Dataset Scale: Processes 463,200 patches extracted from 120 Whole-Slide Images (WSIs).

Preprocessing: Includes automated Macenko stain normalization to eliminate laboratory staining variations alongside a lossless compression pipeline to preserve critical sub-cellular textures.

Architectures Evaluated:
1. Basic ViT
2. Tokens-to-Token ViT (T2T-ViT)
3. CLIP-ViT
4. Swin Transformer (Shifted-Window Attention)
5. Hierarchical ViT (Multi-Scale Feature Fusion)

Explainable AI (XAI): Built-in Grad-CAM visualization suite to map internal attention weights against expert-verified morphological landmarks.

## Results
Our study exposes critical architectural trade-offs when translating vision transformers into high-throughput clinical pathology workflows:

| Architecture | Peak Accuracy | AUROC | Throughput (it/s) | Clinical Strengths |
| :--- | :---: | :---: | :---: | :--- |
| **Hierarchical ViT** | **0.82** | 0.89 | 1.78 | Best for Definitive Diagnosis (Multi-scale translation) |
| **Swin Transformer** | 0.80 | **0.90** | **5.16** | Best for Risk Stratification & Speed ($p = 0.040$) |
| **Basic ViT** | 0.74 | 0.81 | 3.10 | Baseline performance |

### Statistical Note: 
DeLong's test confirmed the Swin Transformer maintains a narrow but statistically significant edge ($p = 0.040$) in global probability separation, alongside a 3x processing speed advantage over Hierarchical ViT. However, Hierarchical ViT delivers the highest absolute hard-label precision.

### Interpretability Mapping
Qualitative Grad-CAM attention tracking confirms that high-performing configurations focus heavily on verified cellular landmarks:
1. Nuclear pleomorphism
2. Chromatin reorganization

## Ethical Approval
This study was conducted with formal ethical clearance granted by the Jawatankuasa Etika Penyelidikan Manusia Universiti Sains Malaysia (Protocol Code: USM/JEPeM/22110749).

## Acknowledgements

Supported by:

1. Fundamental Research Grant Scheme (FRGS), Ministry of Higher Education Malaysia
2. Universiti Sains Malaysia RU Top Down Grant

---

## Contact
Corresponding Author:

Chee Chin Lim

Faculty of Electronic Engineering & Technology

Universiti Malaysia Perlis

Email: cclim@unimap.edu.my
