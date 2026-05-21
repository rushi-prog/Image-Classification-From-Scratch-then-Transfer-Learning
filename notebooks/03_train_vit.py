# %% [markdown]
# # 🤖 Tier 1 — ViT Fine-Tuning on Food-101
# 
# **Run in Colab with T4 GPU.**
# 
# Fine-tunes a Vision Transformer (vit_small_patch16_224) with:
# - Layer-wise Learning Rate Decay (LLRD)
# - Higher Mixup alpha (DeiT recipe)
# - Stochastic depth regularization

# %% [markdown]
# ## 1. Setup

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# Clone the repo
!git clone https://github.com/rushi-prog/Image-Classification-From-Scratch-then-Transfer-Learning.git /content/tier1
%cd /content/tier1

# %%
!pip install -q torch torchvision timm wandb pyyaml tqdm scikit-learn matplotlib seaborn grad-cam albumentations

# %%
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU!'}")

# %%
import wandb
wandb.login()

# %% [markdown]
# ## 2. Train ViT

# %%
# ViT uses batch_size=32 (more memory than CNNs)
# On T4 (16GB), this should fit comfortably
!python scripts/train.py --config configs/vit_finetune.yaml

# %% [markdown]
# ## 3. Evaluate

# %%
!python scripts/evaluate.py \
    --config configs/vit_finetune.yaml \
    --checkpoint results/checkpoints/vit_finetune_food101_best.pth

# %% [markdown]
# ## 4. Grad-CAM / Attention Visualization

# %%
!python scripts/visualize_gradcam.py \
    --config configs/vit_finetune.yaml \
    --checkpoint results/checkpoints/vit_finetune_food101_best.pth \
    --num-images 16

# %%
from IPython.display import Image, display
import glob
for fig in sorted(glob.glob("results/figures/*vit*")):
    print(fig)
    display(Image(fig))

# %%
# Save to Drive
!cp -r results/ /content/drive/MyDrive/tier1_results/vit/
print("Done!")
