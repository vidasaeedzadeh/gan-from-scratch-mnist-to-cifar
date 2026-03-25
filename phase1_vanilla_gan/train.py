"""
phase1_vanilla_gan/train.py

Training loop for Vanilla GAN on MNIST.

GAN Training Logic (alternating steps):
-----------------------------------------
Step 1 — Train Discriminator:
    - Feed REAL images → D should output 1
    - Feed FAKE images (from G) → D should output 0
    - Loss: BCE(D(real), 1) + BCE(D(G(z)), 0)
    - Update D weights

Step 2 — Train Generator:
    - Feed FAKE images to D → G wants D to output 1
    - Loss: BCE(D(G(z)), 1)
    - Update G weights only (D weights frozen)
"""

import os
import sys
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Allow imports from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import get_mnist_loader
from utils.visualize import save_image_grid, plot_losses
from model import Generator, Discriminator


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train():
    cfg = load_config()

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Hyperparameters ──────────────────────────────────────────────────────
    latent_dim  = cfg["model"]["latent_dim"]
    image_size  = cfg["model"]["image_size"]
    channels    = cfg["model"]["channels"]
    epochs      = cfg["training"]["epochs"]
    batch_size  = cfg["training"]["batch_size"]
    lr          = cfg["training"]["lr"]
    beta1       = cfg["training"]["beta1"]
    beta2       = cfg["training"]["beta2"]
    sample_interval = cfg["logging"]["sample_interval"]
    output_dir  = cfg["logging"]["output_dir"]
    ckpt_dir    = cfg["logging"]["checkpoint_dir"]
    data_root   = cfg["data"]["root"]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    loader = get_mnist_loader(batch_size=batch_size, image_size=image_size, data_root=data_root)

    # ── Models ───────────────────────────────────────────────────────────────
    G = Generator(latent_dim=latent_dim, image_size=image_size, channels=channels).to(device)
    D = Discriminator(image_size=image_size, channels=channels).to(device)
    print(f"Generator params:     {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    # ── Loss & Optimizers ────────────────────────────────────────────────────
    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

    # Fixed noise to track Generator progress across epochs
    fixed_noise = torch.randn(64, latent_dim, device=device)

    g_losses, d_losses = [], []

    # ── Training Loop ────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        G.train()
        D.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for real_imgs, _ in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            real_imgs = real_imgs.to(device)
            batch = real_imgs.size(0)

            # Labels
            real_labels = torch.ones(batch, 1, device=device)
            fake_labels = torch.zeros(batch, 1, device=device)

            # ── Step 1: Train Discriminator ──────────────────────────────────
            opt_D.zero_grad()

            # Real images → D should say 1
            d_real = D(real_imgs)
            loss_d_real = criterion(d_real, real_labels)

            # Fake images → D should say 0
            z = torch.randn(batch, latent_dim, device=device)
            fake_imgs = G(z).detach()  # detach so we don't backprop into G here
            d_fake = D(fake_imgs)
            loss_d_fake = criterion(d_fake, fake_labels)

            loss_D = (loss_d_real + loss_d_fake) / 2
            loss_D.backward()
            opt_D.step()

            # ── Step 2: Train Generator ──────────────────────────────────────
            opt_G.zero_grad()

            z = torch.randn(batch, latent_dim, device=device)
            fake_imgs = G(z)
            d_fake_for_g = D(fake_imgs)
            # G wants D to think fake images are real
            loss_G = criterion(d_fake_for_g, real_labels)
            loss_G.backward()
            opt_G.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()

        avg_g = epoch_g_loss / len(loader)
        avg_d = epoch_d_loss / len(loader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)
        print(f"Epoch [{epoch}/{epochs}]  G Loss: {avg_g:.4f}  D Loss: {avg_d:.4f}")

        # ── Save sample images ───────────────────────────────────────────────
        if epoch % sample_interval == 0 or epoch == 1:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise)
            save_image_grid(
                samples,
                path=os.path.join(output_dir, f"epoch_{epoch:03d}.png"),
                nrow=8,
                title=f"Vanilla GAN — Epoch {epoch}",
            )

    # ── Save final checkpoint & loss plot ────────────────────────────────────
    torch.save(G.state_dict(), os.path.join(ckpt_dir, "generator_final.pth"))
    torch.save(D.state_dict(), os.path.join(ckpt_dir, "discriminator_final.pth"))
    plot_losses(g_losses, d_losses, path=os.path.join(output_dir, "losses.png"))
    print("Training complete!")


if __name__ == "__main__":
    train()
