"""
YAML Configuration Loader
===========================
Loads experiment configs from YAML files with defaults,
validation, and dot-notation access.
"""

import os
import yaml
from typing import Any, Optional
from copy import deepcopy


class Config:
    """
    Configuration object with dot-notation access.

    Example:
        config = Config.from_yaml("configs/cnn_scratch.yaml")
        print(config.model.type)          # "cnn_scratch"
        print(config.training.epochs)     # 100
        print(config["data"]["batch_size"])  # Also works with dict access
    """

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
        self._data = data

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def to_dict(self) -> dict:
        """Convert back to a plain dict (recursive)."""
        return deepcopy(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __contains__(self, key):
        return key in self._data

    def __repr__(self):
        return f"Config({self._data})"


def load_config(path: str) -> dict:
    """
    Load a YAML config file and return as a plain dict.

    This is the primary interface — returns a dict for maximum
    compatibility with the rest of the codebase.

    Args:
        path: Path to the YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    # Validate required sections
    _validate_config(config)

    return config


def _validate_config(config: dict) -> None:
    """Validate that required config sections exist."""
    required_sections = ["data", "model", "training"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(
            f"Config missing required sections: {missing}. "
            f"Required: {required_sections}"
        )

    # Validate data section
    data = config["data"]
    if "image_size" not in data:
        raise ValueError("Config 'data' section must specify 'image_size'")
    if "num_classes" not in data:
        raise ValueError("Config 'data' section must specify 'num_classes'")


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep merge two configs. Override values take precedence.
    Useful for CLI argument overrides.

    Args:
        base: Base configuration
        override: Override values

    Returns:
        Merged configuration
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
