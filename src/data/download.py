"""
Dataset Download Utility
========================
Standalone script to pre-download Food-101 dataset.
Useful to run once before training to avoid download during training.
"""

import os
import sys
from torchvision.datasets import Food101


def download_food101(data_dir: str = "./data") -> None:
    """
    Download Food-101 dataset to the specified directory.

    Args:
        data_dir: Root directory for data storage
    """
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("📥 Downloading Food-101 Dataset")
    print("=" * 60)
    print(f"   Directory: {os.path.abspath(data_dir)}")
    print(f"   Size: ~5 GB (compressed)")
    print()

    # Download train split
    print("⏳ Downloading training split...")
    Food101(root=data_dir, split="train", download=True)
    print("✅ Training split ready.")

    # Download test split
    print("⏳ Downloading test split...")
    Food101(root=data_dir, split="test", download=True)
    print("✅ Test split ready.")

    print()
    print("=" * 60)
    print("🎉 Food-101 dataset downloaded successfully!")
    print(f"   Location: {os.path.abspath(data_dir)}")
    print(f"   Training images: 75,750")
    print(f"   Test images: 25,250")
    print(f"   Classes: 101")
    print("=" * 60)


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    download_food101(data_dir)
