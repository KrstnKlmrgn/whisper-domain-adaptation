"""
Utility functions for experiment setup and reproducibility.
"""


import torch
from transformers import enable_full_determinism, set_seed


def set_random_seed(seed):
    """
    Set seeds for reproducible experiments.
    """

    set_seed(seed)

    # Ensure deterministic behavior of PyTorch operations.
    enable_full_determinism(seed)