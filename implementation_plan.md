# 🖼️ Tier 1 — Image Classification: From Scratch → Transfer Learning

Build the definitive "I know fundamentals cold" project — CNN from scratch, Transfer Learning, ViT fine-tuning, Grad-CAM, Mixup/CutMix, W&B tracking.

---

## 🖥️ vs ☁️ — What Runs Where?

> [!IMPORTANT]
> **You don't have a local GPU.** Here's the clean split:

| Where | What | Why |
|-------|------|-----|
| **💻 Local (this workspace)** | Project structure, configs, data pipeline code, utility modules, inference scripts, Grad-CAM visualization, README, results analysis | No GPU needed — code authoring + lightweight CPU work |
| **☁️ Colab (free T4 GPU)** | All training — CNN scratch, ResNet/EfficientNet fine-tune, ViT fine-tune, Mixup/CutMix experiments | Needs GPU for training (30-60 min per experiment on T4) |
| **☁️ Colab (optional)** | W&B dashboard setup (runs during training), final Grad-CAM batch generation on test set | Convenient to run alongside training |

### Colab Workflow
1. Push code to GitHub from local
2. Clone/pull in Colab notebook
3. `pip install -r requirements.txt` in Colab
4. Run training scripts → logs to W&B automatically
5. Download checkpoints + results back to local for analysis

---

## 📁 Project Structure

```
tier1/
├── README.md                    # Project documentation with results
├── requirements.txt             # All dependencies
├── setup.py                     # Package setup (optional)
├── .gitignore
│
├── configs/                     # Experiment configs (YAML)
│   ├── cnn_scratch.yaml         # CNN from scratch config
│   ├── resnet50_finetune.yaml   # ResNet-50 transfer learning
│   ├── efficientnet_finetune.yaml
│   └── vit_finetune.yaml        # ViT fine-tuning config
│
├── src/                         # Core source code
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_scratch.py       # Custom CNN architecture
│   │   ├── transfer_learn.py    # ResNet/EfficientNet wrapper
│   │   └── vit_finetune.py      # ViT fine-tuning wrapper
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset + DataLoader setup
│   │   ├── augmentations.py     # Standard + Mixup/CutMix
│   │   └── download.py          # Dataset download utility
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Unified training loop
│   │   ├── losses.py            # CrossEntropy + label smoothing
│   │   ├── schedulers.py        # LR schedulers (Cosine, OneCycle)
│   │   └── callbacks.py         # Early stopping, checkpointing
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Accuracy, F1, confusion matrix
│   │   └── gradcam.py           # Grad-CAM implementation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # YAML config loader
│       ├── logging_utils.py     # W&B integration
│       └── visualization.py     # Plot utilities
│
├── notebooks/                   # Colab notebooks
│   ├── 01_train_cnn_scratch.ipynb
│   ├── 02_train_transfer_learning.ipynb
│   ├── 03_train_vit.ipynb
│   └── 04_gradcam_analysis.ipynb
│
├── scripts/                     # CLI entry points
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation + metrics
│   └── visualize_gradcam.py     # Grad-CAM visualization
│
├── results/                     # Generated outputs (gitignored, large files)
│   ├── checkpoints/             # Saved model weights
│   ├── figures/                 # Grad-CAM images, plots
│   └── metrics/                 # CSV/JSON results
│
└── assets/                      # README images, architecture diagrams
```

---

## 🗓️ Implementation Phases

### Phase 1 — Foundation (Local) `[~2 hours]`
Set up project structure, configs, data pipeline, augmentation code.

#### [NEW] Core Files
- `requirements.txt` — torch, torchvision, timm, wandb, grad-cam, pyyaml, matplotlib, seaborn, scikit-learn, tqdm, albumentations
- `.gitignore` — checkpoints, data, __pycache__, .env, wandb/
- All `configs/*.yaml` — hyperparameters for each experiment
- `src/data/dataset.py` — CIFAR-10 download + DataLoader factory
- `src/data/augmentations.py` — Standard transforms + **Mixup** + **CutMix** implementation
- `src/utils/config.py` — YAML loader with defaults
- `src/utils/logging_utils.py` — W&B init, log metrics, log images

---

### Phase 2 — CNN From Scratch (Local code → Colab train) `[~3 hours]`
Build a custom CNN, write the training loop, train on Colab.

#### [NEW] Model + Training
- `src/models/cnn_scratch.py` — Custom CNN: 
  - Conv blocks with BatchNorm + ReLU + MaxPool
  - Dropout for regularization
  - Global Average Pooling → FC classifier
  - ~2-5M parameters (deliberately constrained to show you understand architecture choices)
- `src/training/trainer.py` — Unified train loop with:
  - Mixed precision (torch.cuda.amp)
  - Gradient clipping
  - W&B logging per epoch
  - Mixup/CutMix integration
- `src/training/losses.py` — CrossEntropyLoss + LabelSmoothingCE
- `src/training/schedulers.py` — CosineAnnealingWarmRestarts, OneCycleLR
- `src/training/callbacks.py` — EarlyStopping, ModelCheckpoint
- `scripts/train.py` — CLI entry: `python scripts/train.py --config configs/cnn_scratch.yaml`

**🎯 Target**: 88-92% accuracy on CIFAR-10 with scratch CNN.

---

### Phase 3 — Transfer Learning (Local code → Colab train) `[~2 hours]`
Fine-tune ResNet-50 and EfficientNet-B0 with proper unfreezing strategy.

#### [NEW] Transfer Learning Module
- `src/models/transfer_learn.py` — Wrapper that:
  - Loads pretrained backbone (ResNet50, EfficientNetB0)
  - Replaces classifier head
  - Supports freeze/unfreeze strategies (feature extraction → gradual unfreezing → full fine-tune)
  - Discriminative learning rates (lower LR for early layers)
- Configs: `configs/resnet50_finetune.yaml`, `configs/efficientnet_finetune.yaml`

**🎯 Target**: 95-97% accuracy on CIFAR-10 with transfer learning.

---

### Phase 4 — ViT Fine-Tuning (Local code → Colab train) `[~2 hours]`
Fine-tune a Vision Transformer using `timm`.

#### [NEW] ViT Module
- `src/models/vit_finetune.py` — ViT fine-tuning wrapper using `timm`:
  - Load `vit_base_patch16_224` (or `vit_small_patch16_224` for faster training)
  - Image resize to 224×224
  - Layer-wise LR decay
- Config: `configs/vit_finetune.yaml`

**🎯 Target**: 96-98% accuracy on CIFAR-10 with ViT.

---

### Phase 5 — Evaluation + Grad-CAM (Local + Colab) `[~2 hours]`
Build comprehensive evaluation and explainability.

#### [NEW] Evaluation + Visualization
- `src/evaluation/metrics.py` — Accuracy, per-class accuracy, F1, confusion matrix, classification report
- `src/evaluation/gradcam.py` — Grad-CAM implementation:
  - Works with CNN (target last conv layer) and ViT (attention rollout alternative)
  - Generates heatmap overlays on input images
  - Batch processing for test set samples
- `src/utils/visualization.py` — Plotting utilities:
  - Training curves (loss, accuracy)
  - Confusion matrix heatmap
  - Grad-CAM grid visualization
  - Augmentation preview (show Mixup/CutMix examples)
- `scripts/evaluate.py` — Load checkpoint, run full eval, save results
- `scripts/visualize_gradcam.py` — Generate Grad-CAM figures

---

### Phase 6 — Colab Notebooks (Local authoring) `[~1 hour]`
Create clean, well-documented notebooks for Colab execution.

#### [NEW] Notebooks
- `notebooks/01_train_cnn_scratch.ipynb` — Mount Drive, install deps, clone repo, train CNN, log to W&B
- `notebooks/02_train_transfer_learning.ipynb` — ResNet50 + EfficientNet training
- `notebooks/03_train_vit.ipynb` — ViT fine-tuning
- `notebooks/04_gradcam_analysis.ipynb` — Load all 3 checkpoints, generate Grad-CAM comparisons

Each notebook includes:
- One-click Colab GPU setup cells
- `!pip install -r requirements.txt`
- Clone from GitHub
- Training with W&B logging
- Save checkpoints to Google Drive

---

### Phase 7 — README + Results Dashboard `[~1 hour]`
Polish the README with architecture diagrams, results tables, Grad-CAM samples.

#### [NEW] Documentation
- `README.md` with:
  - Project overview + motivation
  - Architecture diagrams (CNN scratch vs pretrained)
  - Results comparison table (all 4 models)
  - Grad-CAM visualizations (side-by-side)
  - W&B dashboard screenshot/link
  - How to reproduce (local + Colab)
  - Key learnings + interview talking points

---

## 📊 Dataset Choice

> [!IMPORTANT]
> **CIFAR-10** for all experiments. Why?
> - Fast iteration (50K train / 10K test, 32×32 images)
> - Perfect for showing scratch CNN vs transfer learning gap
> - Industry-standard benchmark — interviewers know the numbers
> - Small enough for Colab free tier
>
> For transfer learning + ViT, we resize to 224×224 (standard ImageNet input size).

---

## 🔧 Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Dataset** | CIFAR-10 | Fast iteration, known benchmarks, fits Colab free tier |
| **Scratch CNN** | Custom 5-block CNN (~3M params) | Shows architecture design skill, not just API usage |
| **Transfer backbones** | ResNet-50 + EfficientNet-B0 | Classic + modern, good comparison story |
| **ViT** | `vit_small_patch16_224` via timm | Trainable on T4, shows transformer knowledge |
| **Augmentation** | Standard + Mixup + CutMix | Demonstrates advanced data aug understanding |
| **Optimizer** | AdamW (all experiments) | Modern standard, works well with weight decay |
| **LR Schedule** | CosineAnnealing + Warmup | Shows you know training dynamics |
| **Tracking** | Weights & Biases | Industry standard, beautiful dashboards |
| **Explainability** | Grad-CAM | Visual proof your model learns right features |

---

## Open Questions

> [!IMPORTANT]
> 1. **W&B Account**: Do you already have a Weights & Biases account? I'll need your project name to configure logging. (We can also set this up during implementation — it's free.)
> 2. **GitHub Repo**: Should I initialize a git repo here for the Colab workflow? Or are you managing that separately?
> 3. **Food-101 option**: The roadmap mentions Food-101 as an alternative. Want to stick with CIFAR-10 for speed, or use Food-101 for a more impressive portfolio piece? (Food-101 = more training time on Colab, but looks better on resume.)

---

## Verification Plan

### Automated Tests
- Run data pipeline locally: verify DataLoader shapes, augmentation output, Mixup/CutMix correctness
- Smoke test training loop on CPU (1 epoch, tiny batch) before pushing to Colab
- Verify config loading for all 4 experiment configs

### Training Verification (Colab)
- CNN scratch: expect ~90% test accuracy (sanity check)
- ResNet-50 fine-tune: expect ~96% test accuracy
- EfficientNet-B0: expect ~96% test accuracy  
- ViT: expect ~97% test accuracy
- All experiments logged to W&B with loss curves, accuracy, LR schedule

### Visual Verification
- Grad-CAM heatmaps: verify model attends to correct image regions (not background)
- Confusion matrix: identify systematic misclassifications
- Augmentation preview: verify Mixup/CutMix produce valid training samples
