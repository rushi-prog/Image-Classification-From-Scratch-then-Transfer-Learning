"""
Food-101 Dataset & DataLoader Factory
======================================
Downloads Food-101 via torchvision, wraps it with proper transforms,
and returns train/val/test DataLoaders.

Food-101 stats:
- 101 food categories
- 75,750 training images (750 per class)
- 25,250 test images (250 per class)
- Variable image sizes → we resize to config.image_size
"""

import os
from typing import Tuple, Optional, Dict

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import Food101
from PIL import Image

from .augmentations import get_train_transforms, get_val_transforms


# Food-101 class names (alphabetical order as in torchvision)
FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak",
    "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
    "tuna_tartare", "waffles",
]


class Food101Dataset(Dataset):
    """
    Wrapper around torchvision's Food101 that applies transforms.
    Handles downloading, caching, and transform application.
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        transform=None,
        download: bool = True,
    ):
        """
        Args:
            root: Root directory for data storage
            split: 'train' or 'test'
            transform: torchvision transforms to apply
            download: Whether to download if not found
        """
        self.dataset = Food101(
            root=root,
            split=split,
            transform=None,  # We handle transforms ourselves
            download=download,
        )
        self.transform = transform
        self.classes = FOOD101_CLASSES
        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]

        # Ensure RGB (some images might be grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    config: dict,
    val_split: float = 0.1,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders for Food-101.

    The training set is split into train + val (90/10 by default).
    Test set is kept as-is for final evaluation.

    Args:
        config: Configuration dict with data/augmentation settings
        val_split: Fraction of training data to use for validation

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    data_cfg = config["data"]
    aug_cfg = config.get("augmentation", {})

    # Build transforms
    train_transform = get_train_transforms(
        image_size=data_cfg["image_size"],
        aug_config=aug_cfg,
    )
    val_transform = get_val_transforms(
        image_size=data_cfg["image_size"],
        aug_config=aug_cfg,
    )

    # Download and load datasets
    print("📦 Loading Food-101 dataset...")
    full_train_dataset = Food101Dataset(
        root=data_cfg.get("data_dir", "./data"),
        split="train",
        transform=None,  # Transform applied after split
        download=True,
    )
    test_dataset = Food101Dataset(
        root=data_cfg.get("data_dir", "./data"),
        split="test",
        transform=val_transform,
        download=True,
    )

    # Split training into train + validation
    total = len(full_train_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(
            config.get("experiment", {}).get("seed", 42)
        ),
    )

    # Apply transforms to subsets via wrapper
    train_dataset = _TransformSubset(train_subset, train_transform)
    val_dataset = _TransformSubset(val_subset, val_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,  # Important for BatchNorm stability
        persistent_workers=True if data_cfg.get("num_workers", 4) > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=True if data_cfg.get("num_workers", 4) > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=True if data_cfg.get("num_workers", 4) > 0 else False,
    )

    print(f"✅ Dataset loaded:")
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val:   {len(val_dataset)} images")
    print(f"   Test:  {len(test_dataset)} images")
    print(f"   Classes: {data_cfg.get('num_classes', 101)}")
    print(f"   Image size: {data_cfg['image_size']}×{data_cfg['image_size']}")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


class _TransformSubset(Dataset):
    """
    Applies a transform to a Subset (from random_split).
    Needed because random_split returns indices, not a new dataset,
    so we can't set transforms directly on the subset.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # Image comes as PIL from Food101Dataset (transform=None)
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
