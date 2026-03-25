# Phase 2 — DCGAN (Deep Convolutional GAN)

> 🚧 Coming after Phase 1 is complete.

## What's Different from Phase 1
- Generator uses **Transposed Convolutions** (upsampling) instead of Linear layers
- Discriminator uses **Strided Convolutions** (downsampling) instead of Linear layers
- **BatchNorm** added for training stability
- Moves from MNIST (28×28, grayscale) to **CIFAR-10** (32×32, RGB)

## Key Concepts to Learn
- How ConvTranspose2d works (the "deconvolution" layer)
- Why BatchNorm helps GAN training
- The DCGAN architecture guidelines (no pooling, no fully connected layers)

## References
- [DCGAN Paper — Radford et al. 2015](https://arxiv.org/abs/1511.06434)
