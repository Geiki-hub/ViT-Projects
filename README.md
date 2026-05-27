# ViT-Projects
This repository contains the official implementation and benchmarking suite for evaluating Vision Transformer (ViT) architectures to automate MYC status prediction from digital histopathology whole-slide images (WSIs) of Diffuse Large B-Cell Lymphoma (DLBCL).

This pipeline offers a high-throughput, mathematically validated alternative to resource-intensive Fluorescence In Situ Hybridization (FISH) screening by processing standard H&E stained slides.

# Efficiency and Interpretability in MYC Status Prediction: A Comparative Study of Vision Transformers
📌 Project Overview

Identifying MYC status is essential for clinical risk stratification in DLBCL. This project benchmarks five state-of-the-art Vision Transformer configurations across critical diagnostic axes: predictive metrics (Accuracy, AUROC), statistical significance (DeLong's test, 95% CIs), and computational throughput.

[Raw 120 WSIs] ➔ [Macenko Stain Normalization] ➔ [Lossless Compression] ➔ [463,200 Patches] ➔ [ViT Benchmarking]

🛠️ Key Features & Pipeline
Dataset Scale: Processes 463,200 patches extracted from 120 Whole-Slide Images (WSIs).

Preprocessing: Includes automated Macenko stain normalization to eliminate laboratory staining variations alongside a lossless compression pipeline to preserve critical sub-cellular textures.

Architectures Evaluated:
Basic ViT
Tokens-to-Token ViT (T2T-ViT)
CLIP-ViT
Swin Transformer (Shifted-Window Attention)
Hierarchical ViT (Multi-Scale Feature Fusion)

Explainable AI (XAI): Built-in Grad-CAM visualization suite to map internal attention weights against expert-verified morphological landmarks.


📋 Ethical Approval
This study was conducted with formal ethical clearance granted by the Jawatankuasa Etika Penyelidikan Manusia Universiti Sains Malaysia (Protocol Code: USM/JEPeM/22110749).
