# %% [markdown]
# # 🖼️ Tier 1 — Train CNN From Scratch on Food-101
# 
# **Run this notebook in Google Colab with GPU runtime.**
# 
# Runtime → Change runtime type → T4 GPU

# %% [markdown]
# ## 1. Setup

# %%
# Mount Google Drive (to save checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# %%
# Clone the repo
!git clone https://github.com/rushi-prog/Image-Classification-From-Scratch-then-Transfer-Learning.git /content/tier1
%cd /content/tier1

# %%
# Install dependencies
!pip install -q torch torchvision timm wandb pyyaml tqdm scikit-learn matplotlib seaborn grad-cam albumentations

# %%
# Verify GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Login to Weights & Biases

# %%
import wandb
wandb.login()  # Paste your API key when prompted

# %% [markdown]
# ## 3. Train CNN From Scratch

# %%
# Run training!
!python scripts/train.py --config configs/cnn_scratch.yaml

# %% [markdown]
# ## 4. Results
# 
# After training completes:
# - Checkpoints saved to `results/checkpoints/`
# - Training curves saved to `results/figures/`
# - Metrics saved to `results/metrics/`
# - W&B dashboard at your wandb.ai project

# %%
# View training curves
from IPython.display import Image, display
import glob

for fig in glob.glob("results/figures/*training_curves*"):
    print(fig)
    display(Image(fig))

# %%
# Run evaluation on test set
!python scripts/evaluate.py \
    --config configs/cnn_scratch.yaml \
    --checkpoint results/checkpoints/cnn_scratch_food101_best.pth

# %%
# Generate Grad-CAM visualizations
!python scripts/visualize_gradcam.py \
    --config configs/cnn_scratch.yaml \
    --checkpoint results/checkpoints/cnn_scratch_food101_best.pth \
    --num-images 16

# %%
# Display Grad-CAM grid
for fig in glob.glob("results/figures/*gradcam*"):
    print(fig)
    display(Image(fig))

# %%
# Copy checkpoints to Google Drive for safekeeping
!cp -r results/ /content/drive/MyDrive/tier1_results/cnn_scratch/
print("Checkpoints saved to Google Drive!")
