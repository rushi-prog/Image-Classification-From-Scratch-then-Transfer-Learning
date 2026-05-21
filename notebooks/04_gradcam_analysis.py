# %% [markdown]
# # 🔍 Tier 1 — Grad-CAM Analysis: All Models Compared
# 
# Load all 4 trained models and generate side-by-side Grad-CAM comparisons.
# Shows how different architectures "see" the same food images.

# %% [markdown]
# ## 1. Setup

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
!pip install -q torch torchvision timm wandb pyyaml tqdm scikit-learn matplotlib seaborn grad-cam albumentations

# %% [markdown]
# ## 2. Generate Grad-CAM for All Models

# %%
import subprocess
import os

models = [
    ("CNN Scratch", "configs/cnn_scratch.yaml", "results/checkpoints/cnn_scratch_food101_best.pth"),
    ("ResNet-50", "configs/resnet50_finetune.yaml", "results/checkpoints/resnet50_finetune_food101_best.pth"),
    ("EfficientNet-B0", "configs/efficientnet_finetune.yaml", "results/checkpoints/efficientnet_b0_finetune_food101_best.pth"),
    ("ViT-Small", "configs/vit_finetune.yaml", "results/checkpoints/vit_finetune_food101_best.pth"),
]

for name, config, ckpt in models:
    if os.path.exists(ckpt):
        print(f"\n{'='*50}")
        print(f"Generating Grad-CAM for {name}")
        print(f"{'='*50}")
        subprocess.run([
            "python", "scripts/visualize_gradcam.py",
            "--config", config,
            "--checkpoint", ckpt,
            "--num-images", "8"
        ])
    else:
        print(f"Skipping {name} — checkpoint not found: {ckpt}")

# %% [markdown]
# ## 3. Display All Grad-CAM Grids

# %%
from IPython.display import Image, display
import glob

for fig in sorted(glob.glob("results/figures/*gradcam*")):
    print(f"\n{fig}")
    display(Image(fig, width=900))

# %% [markdown]
# ## 4. Model Comparison Summary

# %%
import json
import os

print(f"{'Model':<25} {'Top-1 Acc':>10} {'F1 Macro':>10} {'Params':>10}")
print("-" * 60)

for name, config, ckpt in models:
    exp_name = os.path.basename(config).replace(".yaml", "").replace("_finetune", "")
    metrics_path = f"results/metrics/{exp_name}_eval_results.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        print(f"{name:<25} {m.get('accuracy', 0):>9.2f}% {m.get('f1_macro', 0):>9.2f}% {'':>10}")
    else:
        print(f"{name:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

# %%
# Save everything to Drive
!cp -r results/ /content/drive/MyDrive/tier1_results/final/
print("All results saved to Drive!")
