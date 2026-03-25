# Phase 1 — Vanilla GAN on MNIST

## 🎯 Goal
Build the simplest possible GAN using fully-connected layers (MLP) to generate handwritten digits from MNIST.

## 🧠 What You'll Learn
- The core GAN training loop (Generator vs Discriminator)
- Binary Cross-Entropy (BCE) loss for both networks
- Why training GANs is tricky (mode collapse, vanishing gradients)
- How to inspect and visualize generated samples during training

## 🏗️ Architecture

```
Noise z (latent_dim=100)
        │
   [Generator]          MLP: Linear → LeakyReLU → ... → Tanh
        │
  Fake Image (28x28)
        │
 [Discriminator]        MLP: Linear → LeakyReLU → ... → Sigmoid
        │
  Real or Fake? (scalar)
```

## ▶️ How to Run

```bash
cd phase1_vanilla_gan
python train.py
```

Outputs are saved to `outputs/phase1/`:
- `epoch_XXX.png` — grid of generated images per epoch
- `losses.png` — Generator & Discriminator loss curves

## ⚙️ Config

Edit `configs/config.yaml` to change hyperparameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `latent_dim` | 100 | Size of noise vector fed to Generator |
| `lr` | 0.0002 | Learning rate for both G and D |
| `batch_size` | 128 | Samples per training step |
| `epochs` | 50 | Total training epochs |

## 🔍 What to Observe
- Early epochs: pure noise
- Mid training: recognizable but blurry digits
- Watch for **mode collapse** — G produces only one digit type
- D loss near 0 + G loss rising = D is winning (bad sign)

## 📝 Notes / Observations
*(Write your training observations here as you experiment)*
