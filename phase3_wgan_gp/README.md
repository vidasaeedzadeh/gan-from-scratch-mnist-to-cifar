# Phase 3 — WGAN-GP (Wasserstein GAN with Gradient Penalty)

> 🚧 Coming after Phase 2 is complete.

## What's Different from Phase 2
- Replaces BCE loss with **Wasserstein distance** for more meaningful training signal
- Discriminator becomes a **Critic** (no Sigmoid, outputs unbounded score)
- **Gradient Penalty** replaces weight clipping for Lipschitz constraint
- Much more **stable training** — less mode collapse

## Key Concepts to Learn
- What the Wasserstein distance measures
- Why the original GAN loss suffers from vanishing gradients
- How to implement gradient penalty in PyTorch
- FID score as a quantitative evaluation metric

## References
- [WGAN Paper — Arjovsky et al. 2017](https://arxiv.org/abs/1701.07875)
- [WGAN-GP Paper — Gulrajani et al. 2017](https://arxiv.org/abs/1704.00028)
