"""
phase1_vanilla_gan/evaluate.py

Load a saved Generator checkpoint and generate sample images.
Run this after training to visualize results without re-training.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint ../checkpoints/phase1/generator_final.pth
"""

import os
import sys
import argparse
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.visualize import save_image_grid
from model import Generator


def evaluate(checkpoint: str, latent_dim: int = 100, n_samples: int = 64, output_dir: str = "../outputs/phase1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(latent_dim=latent_dim, image_size=28, channels=1).to(device)
    G.load_state_dict(torch.load(checkpoint, map_location=device))
    G.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        samples = G(z)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "eval_samples.png")
    save_image_grid(samples, path=out_path, nrow=8, title="Vanilla GAN — Evaluation Samples")
    print(f"Saved {n_samples} samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../checkpoints/phase1/generator_final.pth")
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--output_dir", default="../outputs/phase1")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.latent_dim, args.n_samples, args.output_dir)
