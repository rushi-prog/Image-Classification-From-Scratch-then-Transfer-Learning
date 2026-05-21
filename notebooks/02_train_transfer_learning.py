# %% [markdown]
# # 🔄 Tier 1 — Transfer Learning: ResNet-50 + EfficientNet-B0
# 
# **Run in Colab with T4 GPU.**
# 
# This trains two pretrained models on Food-101 with:
# - Gradual unfreezing (train head first, then deeper layers)
# - Discriminative learning rates (backbone 10x lower LR)
# - Mixup + CutMix augmentation

# %% [markdown]
# ## 1. Setup

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
!pip install -q torch torchvision timm wandb pyyaml tqdm scikit-learn matplotlib seaborn grad-cam albumentations

# %%
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU!'}")

# %%
import wandb
wandb.login()

# %% [markdown]
# ## 2. Train ResNet-50

# %%
!python scripts/train.py --config configs/resnet50_finetune.yaml

# %% [markdown]
# ## 3. Train EfficientNet-B0

# %%
!python scripts/train.py --config configs/efficientnet_finetune.yaml

# %% [markdown]
# ## 4. Evaluate Both Models

# %%
# Evaluate ResNet-50
!python scripts/evaluate.py \
    --config configs/resnet50_finetune.yaml \
    --checkpoint results/checkpoints/resnet50_finetune_food101_best.pth

# %%
# Evaluate EfficientNet-B0
!python scripts/evaluate.py \
    --config configs/efficientnet_finetune.yaml \
    --checkpoint results/checkpoints/efficientnet_b0_finetune_food101_best.pth

# %% [markdown]
# ## 5. Grad-CAM Comparison

# %%
!python scripts/visualize_gradcam.py \
    --config configs/resnet50_finetune.yaml \
    --checkpoint results/checkpoints/resnet50_finetune_food101_best.pth \
    --num-images 16

# %%
!python scripts/visualize_gradcam.py \
    --config configs/efficientnet_finetune.yaml \
    --checkpoint results/checkpoints/efficientnet_b0_finetune_food101_best.pth \
    --num-images 16

# %%
from IPython.display import Image, display
import glob
for fig in sorted(glob.glob("results/figures/*gradcam*")):
    print(fig)
    display(Image(fig))

# %%
# Save to Drive
!cp -r results/ /content/drive/MyDrive/tier1_results/transfer_learning/
print("Done!")
