"""
utils/metrics.py
Evaluation metrics — FID score added in Phase 3+.
For now contains a placeholder and basic utilities.
"""

import torch
import numpy as np


def count_parameters(model: torch.nn.Module) -> int:
    """Returns the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, name: str = "Model"):
    """Prints a short summary of a model."""
    n = count_parameters(model)
    print(f"{name}: {n:,} trainable parameters")
    print(model)


# FID score will be implemented in Phase 3
# Placeholder to keep imports consistent across phases
def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    Frechet Inception Distance (FID).
    Lower is better. Will be fully implemented in Phase 3.
    """
    raise NotImplementedError("FID score is implemented in Phase 3 (utils/metrics.py).")
