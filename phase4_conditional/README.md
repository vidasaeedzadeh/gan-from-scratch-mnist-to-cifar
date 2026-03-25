# Phase 4 — Conditional GAN (cGAN)

> 🚧 Coming after Phase 3 is complete.

## What's Different from Phase 3
- Generator and Discriminator both receive a **class label** as input
- Allows **controlled generation** — you choose which class to generate (e.g., "give me a dog")
- Label is embedded and concatenated to the noise vector (G) or image (D)

## Key Concepts to Learn
- Label conditioning via `nn.Embedding`
- How to concatenate conditioning info to noise and images
- Evaluating class-conditional generation quality

## References
- [Conditional GAN — Mirza & Osindero 2014](https://arxiv.org/abs/1411.1784)
