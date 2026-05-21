<div align="center">

# 🖼️ Image Classification: From Scratch → Transfer Learning → ViT

**A complete image classification pipeline on Food-101**

CNN From Scratch • ResNet-50 • EfficientNet-B0 • Vision Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rushi-prog/Image-Classification-From-Scratch-then-Transfer-Learning/blob/main/notebooks/01_train_cnn_scratch.ipynb)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 💡 About

This project progressively builds image classifiers on [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (101 food categories, 101K images) — starting from a CNN designed from scratch, then leveraging transfer learning, and finally fine-tuning a Vision Transformer. Every model includes Grad-CAM explainability, modern augmentation (Mixup/CutMix), and live W&B experiment tracking.

```
CNN From Scratch  ──→  ResNet-50 / EfficientNet-B0  ──→  ViT-Small
   (9.57M params)         (23.7M / 4.1M params)          (21.7M params)
```

---

## 🏗️ Models & Techniques

| Model | Architecture | Key Technique | Params |
|:------|:-------------|:--------------|:------:|
| **ScratchCNN** | Custom 5-block CNN | Kaiming init, BatchNorm, progressive dropout | 9.57M |
| **ResNet-50** | Transfer Learning | Gradual unfreezing + discriminative LR | 23.71M |
| **EfficientNet-B0** | Transfer Learning | Compound scaling, efficient backbone | 4.14M |
| **ViT-Small** | Vision Transformer | Layer-wise LR Decay (LLRD), stochastic depth | 21.70M |

### Training Techniques
- **Mixup & CutMix** — batch-level augmentation with soft labels
- **Label Smoothing** — prevents overconfident predictions
- **Mixed Precision (AMP)** — FP16 training for 2x speed on GPU
- **Cosine Annealing + Warmup** — smooth LR scheduling
- **Gradient Clipping** — training stability
- **EarlyStopping + Checkpointing** — automatic best model saving

### Explainability
- **Grad-CAM** — heatmaps showing what each model focuses on
- Works across CNN, ResNet, EfficientNet, and ViT architectures

---

## 📁 Project Structure

```
├── configs/                          # YAML experiment configs
│   ├── cnn_scratch.yaml
│   ├── resnet50_finetune.yaml
│   ├── efficientnet_finetune.yaml
│   └── vit_finetune.yaml
│
├── src/
│   ├── data/                         # Dataset loading + augmentations
│   │   ├── dataset.py                # Food-101 DataLoader factory
│   │   └── augmentations.py          # Transforms + Mixup + CutMix
│   │
│   ├── models/                       # All model architectures
│   │   ├── cnn_scratch.py            # Custom CNN (from scratch)
│   │   ├── transfer_learn.py         # ResNet/EfficientNet wrapper
│   │   └── vit_finetune.py           # ViT via timm with LLRD
│   │
│   ├── training/                     # Training infrastructure
│   │   ├── trainer.py                # Unified training loop (AMP, W&B)
│   │   ├── losses.py                 # Label smoothing + soft target CE
│   │   ├── schedulers.py             # LR schedulers with warmup
│   │   └── callbacks.py              # EarlyStopping + ModelCheckpoint
│   │
│   ├── evaluation/                   # Evaluation + explainability
│   │   ├── metrics.py                # Top-1/5, F1, per-class accuracy
│   │   └── gradcam.py                # Grad-CAM for all model types
│   │
│   └── utils/                        # Config, logging, visualization
│
├── scripts/                          # CLI entry points
│   ├── train.py                      # python scripts/train.py --config ...
│   ├── evaluate.py                   # Full test set evaluation
│   └── visualize_gradcam.py          # Generate Grad-CAM grids
│
└── notebooks/                        # Colab notebooks (.ipynb)
    ├── 01_train_cnn_scratch.ipynb
    ├── 02_train_transfer_learning.ipynb
    ├── 03_train_vit.ipynb
    └── 04_gradcam_analysis.ipynb
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

Click the badge or open any notebook from `notebooks/` in Colab with **T4 GPU** runtime:

| Notebook | What | Colab |
|----------|------|:-----:|
| `01_train_cnn_scratch.ipynb` | Train CNN from scratch | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rushi-prog/Image-Classification-From-Scratch-then-Transfer-Learning/blob/main/notebooks/01_train_cnn_scratch.ipynb) |
| `02_train_transfer_learning.ipynb` | ResNet-50 + EfficientNet-B0 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rushi-prog/Image-Classification-From-Scratch-then-Transfer-Learning/blob/main/notebooks/02_train_transfer_learning.ipynb) |
| `03_train_vit.ipynb` | ViT fine-tuning with LLRD | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rushi-prog/Image-Classification-From-Scratch-then-Transfer-Learning/blob/main/notebooks/03_train_vit.ipynb) |
| `04_gradcam_analysis.ipynb` | Grad-CAM for all models | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rushi-prog/Image-Classification-From-Scratch-then-Transfer-Learning/blob/main/notebooks/04_gradcam_analysis.ipynb) |

### Option 2: CLI

```bash
# Install
pip install torch torchvision timm wandb pyyaml tqdm scikit-learn matplotlib seaborn

# Train any model
python scripts/train.py --config configs/cnn_scratch.yaml
python scripts/train.py --config configs/resnet50_finetune.yaml
python scripts/train.py --config configs/vit_finetune.yaml

# Override from CLI
python scripts/train.py --config configs/cnn_scratch.yaml --epochs 50 --lr 0.0005 --no-wandb

# Evaluate
python scripts/evaluate.py --config configs/cnn_scratch.yaml --checkpoint results/checkpoints/cnn_scratch_food101_best.pth

# Grad-CAM
python scripts/visualize_gradcam.py --config configs/cnn_scratch.yaml --checkpoint <path> --num-images 16
```

---

## 📊 Results

> _Results table will be updated after training runs._

| Model | Top-1 Acc | Top-5 Acc | F1 (macro) | Params |
|:------|:---------:|:---------:|:----------:|:------:|
| CNN Scratch | — | — | — | 9.57M |
| ResNet-50 | — | — | — | 23.71M |
| EfficientNet-B0 | — | — | — | 4.14M |
| ViT-Small | — | — | — | 21.70M |

---

## 🔍 Grad-CAM Visualizations

> _Will be added after training. Shows what each model "looks at" when classifying food._

<!-- ![Grad-CAM Grid](results/figures/gradcam_comparison.png) -->

---

## 🧠 Design Decisions

<details>
<summary><b>Why Food-101?</b></summary>

- **101 classes** — enough complexity to show architecture matters
- **101K images** — realistic scale, not a toy dataset  
- **Noisy labels** — real-world challenge (some training images are intentionally mislabeled)
- **Fine-grained** — distinguishing "steak" from "filet mignon" requires nuance
</details>

<details>
<summary><b>Why Mixup + CutMix?</b></summary>

- Regularization without extra data
- Forces model to learn from partial information
- Reduces overconfidence (synergistic with label smoothing)
- Used in every modern recipe: DeiT, ResNeXt-WSL, EfficientNetV2
</details>

<details>
<summary><b>Why Layer-wise LR Decay for ViT?</b></summary>

- Shallow ViT layers capture positional/patch features → preserve them
- Deep layers + head need to adapt to the new task → higher LR
- Standard in transformer fine-tuning (BEiT, MAE, DINOv2)
</details>

<details>
<summary><b>Why Gradual Unfreezing for Transfer Learning?</b></summary>

- Prevents catastrophic forgetting of pretrained features
- Train classifier head first (new task), then deeper backbone layers
- Discriminative LR: backbone gets 10x lower learning rate
</details>

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Model building & training |
| torchvision | Pretrained models & transforms |
| timm | Vision Transformer models |
| Weights & Biases | Experiment tracking |
| scikit-learn | Evaluation metrics |
| matplotlib / seaborn | Visualization |

---

## 📄 License

MIT
