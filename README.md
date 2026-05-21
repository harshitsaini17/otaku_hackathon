# Otaku Hackathon

A deep-learning project that classifies **630+ anime character faces** using transfer learning, data augmentation, and advanced visualization techniques. Built for a hackathon with a focus on rigorous exploratory data analysis before training.

## What It Does

Given an anime face image, the model predicts which of 630+ possible characters it belongs to, trained on a dataset of ~93K images. The project includes a comprehensive exploratory data analysis (EDA) pipeline that informs training strategy.

## Key Features

- **630-class character classification** — wide-coverage anime face recognition
- **Transfer learning backbone** — TIMM pretrained models with fine-tuning
- **Data augmentation pipeline** — Albumentations with configurable transforms
- **Class imbalance detection** — automatic WeightedRandomSampler recommendation
- **Advanced visualizations** — class distribution, resolution analysis, UMAP embedding plots
- **GPU-accelerated training** — with automatic mixed precision (AMP)
- **Stratified training/validation split** — preserving class balance

## Project Structure

```
otaku-hackathon/
├── preprocessing.ipynb     # Full EDA + training + evaluation pipeline
├── dataset/
│   ├── train/              # Training images
│   ├── test/               # Test images
│   └── label_mapping.csv   # 630-class label → name mapping
└── working/                # Generated plots & experiment artifacts
```

## Workflow

The notebook is organized into analytical sections:

| Section | Analysis | Purpose |
|---------|----------|---------|
| Section 1 | Class Distribution Histogram | Detects class imbalance and recommends `WeightedRandomSampler` / class-weighted loss |
| Section 2 | Image Resolution & Aspect Ratio | Confirms safe crop targets and `RandomResizedCrop` scale ranges |
| Section 3 | Pixel Intensity Distribution | Informs data normalization & augmentation strategies |
| Section 4 | Sample Qualitative Inspection | Manual sanity check for data quality issues |
| Section 5 | UMAP Embedding Visualization | Validates that the model learns semantically meaningful clusters |
| Training | Transfer Learning + Fine-tuning | End-to-end training with AMP, early stopping, LR scheduling |
| Evaluation | Confusion Matrix + Per-Class F1 | Identifies hard classes for targeted augmentation |

## Tech Stack

- **Python 3.12**
- **PyTorch** — deep learning framework
- **TIMM** — pretrained model zoo (EfficientNet, ResNet, ViT backbones)
- **Albumentations** — advanced image augmentation
- **UMAP** — dimensionality reduction for embedding visualization
- **Seaborn / Matplotlib** — exploratory plotting

## Setup

```bash
# Install dependencies
pip install torch torchvision umap-learn albumentations timm seaborn matplotlib pandas pillow opencv-python

# Run the notebook
jupyter lab preprocessing.ipynb
```

## Results

- **Total images:** ~93,145
- **Train/test split:** 80/20 stratified
- **Backbone:** TIMM EfficientNet-B0 (configurable)
- **Final input size:** 336×336
- **Evaluation metric:** Macro F1 (critical for long-tail class performance)

## Learning Highlights

- Rigorous EDA-first approach: every training hyperparameter is informed by data analysis
- UMAP visualization as a quality gate: ensures embeddings are semantically meaningful before deployment
- Class imbalance handling: automatic `WeightedRandomSampler` recommendation based on imbalance ratio

## Acknowledgments

Built for a hackathon focused on anime character recognition. The label mapping supports 630+ characters spanning dozens of popular anime series.
