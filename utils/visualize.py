"""
utils/visualize.py
Shared visualization utilities: image grids, loss curves, training GIFs.
"""

import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
import numpy as np


def save_image_grid(tensor: torch.Tensor, path: str, nrow: int = 8, title: str = None):
    """
    Save a grid of generated images to disk.
    tensor: (N, C, H, W) in range [-1, 1]
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1).cpu()

    grid = vutils.make_grid(tensor, nrow=nrow, padding=2)
    np_grid = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    if title:
        plt.title(title, fontsize=14)
    plt.imshow(np_grid, cmap="gray" if np_grid.shape[-1] == 1 else None)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")


def plot_losses(g_losses: list, d_losses: list, path: str):
    """
    Plot Generator and Discriminator loss curves and save to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss", alpha=0.8)
    plt.plot(d_losses, label="Discriminator Loss", alpha=0.8)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Saved] {path}")
