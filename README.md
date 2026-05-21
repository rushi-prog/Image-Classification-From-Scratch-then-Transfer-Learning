# 🖼️ Tier 1 — Image Classification: From Scratch → Transfer Learning → ViT

> "I know the fundamentals cold."

A complete image classification pipeline on **Food-101** (101 food categories, 101K images), progressing from a CNN built from scratch to Vision Transformer fine-tuning — with Grad-CAM explainability, Mixup/CutMix augmentation, and W&B experiment tracking.

---

## 🏗️ Architecture Progression

```
CNN From Scratch  →  ResNet-50 / EfficientNet-B0  →  ViT-Small
   (9.57M params)       (23.7M / 4.1M params)        (21.7M params)
   ~45-55% acc          ~78-85% acc                   ~82-88% acc
```

| Model | Type | Key Technique | Expected Acc |
|-------|------|---------------|:---:|
| **ScratchCNN** | 5-block CNN | Kaiming init, progressive dropout | ~50% |
| **ResNet-50** | Transfer Learning | Gradual unfreezing, discriminative LR | ~82% |
| **EfficientNet-B0** | Transfer Learning | Compound scaling, efficient backbone | ~80% |
| **ViT-Small** | Transformer | Layer-wise LR decay (LLRD) | ~85% |

---

## 🔥 What Makes This Portfolio-Grade

### Techniques Implemented
- **Mixup & CutMix** — batch-level augmentation with soft label support
- **Label Smoothing** — prevents overconfident predictions
- **Gradual Unfreezing** — train head first, then progressively unfreeze backbone
- **Discriminative Learning Rates** — backbone gets 10x lower LR than head
- **Layer-wise LR Decay (LLRD)** — ViT-specific: deeper layers get higher LR
- **Mixed Precision (AMP)** — FP16 training for 2x GPU speedup
- **Gradient Clipping** — prevents exploding gradients
- **Cosine Annealing + Warm Restarts** — LR scheduling with warmup
- **Grad-CAM** — visual explainability for all model types

### Engineering Quality
- **Unified Trainer** — one training loop handles CNN, ResNet, EfficientNet, ViT
- **Config-driven** — YAML configs control everything (no hardcoded values)
- **W&B Dashboard** — real-time training monitoring
- **EarlyStopping + Checkpointing** — automatic best model saving
- **CLI with overrides** — `--epochs 50 --lr 0.0005 --no-wandb`

---

## 📁 Project Structure

```
tier1/
├── configs/                        # Experiment configs (YAML)
│   ├── cnn_scratch.yaml            # CNN from scratch
│   ├── resnet50_finetune.yaml      # ResNet-50 transfer learning
│   ├── efficientnet_finetune.yaml  # EfficientNet-B0 transfer learning
│   └── vit_finetune.yaml          # ViT fine-tuning
│
├── src/                            # Source code
│   ├── data/
│   │   ├── dataset.py              # Food-101 DataLoader factory
│   │   ├── augmentations.py        # Transforms + Mixup + CutMix
│   │   └── download.py             # Dataset download script
│   │
│   ├── models/
│   │   ├── cnn_scratch.py          # Custom 5-block CNN (9.57M params)
│   │   ├── transfer_learn.py       # ResNet/EfficientNet wrapper
│   │   └── vit_finetune.py         # ViT via timm with LLRD
│   │
│   ├── training/
│   │   ├── trainer.py              # Unified training loop
│   │   ├── losses.py               # Label smoothing + soft target CE
│   │   ├── schedulers.py           # Cosine + warmup schedulers
│   │   └── callbacks.py            # EarlyStopping + ModelCheckpoint
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # Top-1/5, F1, per-class accuracy
│   │   └── gradcam.py              # Grad-CAM + heatmap overlay
│   │
│   └── utils/
│       ├── config.py               # YAML config loader
│       ├── logging_utils.py        # W&B logger
│       └── visualization.py        # Training curves, confusion matrix
│
├── scripts/                        # CLI entry points
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   └── visualize_gradcam.py        # Grad-CAM generation
│
├── notebooks/                      # Colab notebooks (.py format)
│   ├── 01_train_cnn_scratch.py
│   ├── 02_train_transfer_learning.py
│   ├── 03_train_vit.py
│   └── 04_gradcam_analysis.py
│
└── results/                        # Training outputs (gitignored)
    ├── checkpoints/                # Model weights
    ├── figures/                    # Training curves, Grad-CAM
    └── metrics/                    # JSON evaluation results
```

---

## 🚀 Quick Start

### Local (CPU — for testing)
```bash
# Install dependencies
uv add torch torchvision timm wandb pyyaml tqdm scikit-learn matplotlib seaborn

# Run smoke tests
uv run python test_phase2.py
uv run python test_phase3_5.py
```

### Google Colab (GPU — for training)
1. Upload the project to Google Drive or clone from GitHub
2. Open any notebook from `notebooks/` in Colab
3. Set runtime to **T4 GPU**
4. Run all cells

### CLI Training
```bash
# Train CNN from scratch
python scripts/train.py --config configs/cnn_scratch.yaml

# Train ResNet-50 (transfer learning)
python scripts/train.py --config configs/resnet50_finetune.yaml

# Train EfficientNet-B0
python scripts/train.py --config configs/efficientnet_finetune.yaml

# Train ViT
python scripts/train.py --config configs/vit_finetune.yaml

# With overrides
python scripts/train.py --config configs/cnn_scratch.yaml --epochs 50 --lr 0.0005 --no-wandb
```

### Evaluation
```bash
# Full test evaluation
python scripts/evaluate.py \
    --config configs/cnn_scratch.yaml \
    --checkpoint results/checkpoints/cnn_scratch_food101_best.pth

# Grad-CAM visualization
python scripts/visualize_gradcam.py \
    --config configs/cnn_scratch.yaml \
    --checkpoint results/checkpoints/cnn_scratch_food101_best.pth \
    --num-images 16
```

---

## 📊 Results

> Results will be populated after training runs on Colab.

| Model | Top-1 Acc | Top-5 Acc | F1 (macro) | Params | Training Time |
|-------|:---------:|:---------:|:----------:|:------:|:------------:|
| CNN Scratch | — | — | — | 9.57M | — |
| ResNet-50 | — | — | — | 23.71M | — |
| EfficientNet-B0 | — | — | — | 4.14M | — |
| ViT-Small | — | — | — | 21.70M | — |

---

## 🔍 Grad-CAM Visualizations

> Will be generated after training. Shows what each model "looks at" when classifying food.

```bash
python scripts/visualize_gradcam.py --config configs/cnn_scratch.yaml --checkpoint <path> --num-images 16
```

---

## 🧠 Key Design Decisions

### Why Food-101?
- 101 classes — enough complexity to show model architecture matters
- 101K images — realistic scale, not a toy dataset
- Noisy labels — real-world challenge (some images are intentionally mislabeled)
- Fine-grained recognition — distinguishing "steak" from "filet mignon" requires nuance

### Why Mixup + CutMix?
- Regularization without extra data
- Forces the model to learn from partial information
- Reduces overconfidence (works synergistically with label smoothing)
- Used in every modern training recipe (DeiT, ResNeXt-WSL, EfficientNetV2)

### Why Layer-wise LR Decay for ViT?
- ViT's shallow layers capture positional and patch-level features
- Deep layers + head need to adapt to Food-101
- LLRD preserves pretrained representations while allowing task adaptation
- Standard in transformer fine-tuning (BEiT, MAE, DINOv2)

---

## 📝 Interview Prep Points

<details>
<summary>Click to expand key talking points</summary>

### CNN From Scratch
- **Kaiming initialization** — proper init for ReLU networks prevents vanishing/exploding gradients
- **Batch normalization** — normalizes layer inputs, allows higher LR, mild regularization
- **Global Average Pooling** — replaces FC layers, reduces parameters, adds spatial invariance

### Transfer Learning
- **Why it works** — ImageNet features (edges, textures, shapes) transfer to food images
- **Gradual unfreezing** — prevents catastrophic forgetting of pretrained features
- **Discriminative LR** — early layers need less change (universal features), later layers need more (task-specific)

### Vision Transformers
- **Self-attention** — captures global dependencies (a CNN needs many stacked layers for this)
- **Positional embeddings** — ViT has no inductive bias for locality, needs explicit position info
- **LLRD** — critical for fine-tuning: deeper layers adapted more, shallower layers preserved
- **ViT needs more data** — DeiT showed regularization (Mixup, CutMix, stochastic depth) can compensate

### Grad-CAM
- **Uses gradients** — gradient of target class w.r.t. feature maps shows importance
- **Channel weights** — global average pool of gradients gives per-channel importance
- **Validates correctness** — model should look at the food, not the plate or background

</details>

---

## 🛠️ Tech Stack

- **PyTorch** — model building and training
- **torchvision** — pretrained models and transforms
- **timm** — Vision Transformer models
- **Weights & Biases** — experiment tracking and dashboards
- **scikit-learn** — evaluation metrics
- **matplotlib / seaborn** — visualization

---

## 📄 License

MIT
