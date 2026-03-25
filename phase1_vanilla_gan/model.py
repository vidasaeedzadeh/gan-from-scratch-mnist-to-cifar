"""
phase1_vanilla_gan/model.py

Vanilla GAN: Generator and Discriminator as Multi-Layer Perceptrons (MLP).
Both networks are intentionally simple — the goal is to understand the
core GAN dynamic before adding convolutional layers in Phase 2.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Maps a random noise vector z ~ N(0, 1) to a fake image.

    Input:  (batch_size, latent_dim)
    Output: (batch_size, channels * image_size * image_size)
            reshaped to (batch_size, channels, image_size, image_size) in train.py

    Activation: Tanh on the output layer so pixel values are in [-1, 1],
                matching the normalized real images.
    """

    def __init__(self, latent_dim: int = 100, image_size: int = 28, channels: int = 1):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        output_dim = channels * image_size * image_size  # 784 for MNIST

        self.net = nn.Sequential(
            # Block 1: latent_dim → 256
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: 256 → 512
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: 512 → 1024
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer
            nn.Linear(1024, output_dim),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        # Reshape flat vector to image: (batch, C, H, W)
        return out.view(out.size(0), self.channels, self.image_size, self.image_size)


class Discriminator(nn.Module):
    """
    Classifies images as real (1) or fake (0).

    Input:  (batch_size, channels, image_size, image_size) — will be flattened
    Output: (batch_size, 1) — probability of being real

    Note: LeakyReLU (not ReLU) is standard in GANs to avoid dead neurons.
    Dropout adds regularization so D doesn't overpower G too quickly.
    """

    def __init__(self, image_size: int = 28, channels: int = 1):
        super().__init__()
        input_dim = channels * image_size * image_size  # 784 for MNIST

        self.net = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Block 2
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Block 3
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Output: single probability
            nn.Linear(256, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Flatten image to vector
        x = img.view(img.size(0), -1)
        return self.net(x)
