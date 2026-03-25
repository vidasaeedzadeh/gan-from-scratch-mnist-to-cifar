# Theory — Vanilla GAN

## The Core Idea

A GAN consists of two networks locked in competition:

- **Generator (G)**: Takes random noise `z` and generates fake data. Its job is to fool D.
- **Discriminator (D)**: Sees both real and fake data. Its job is to tell them apart.

They play a **minimax game** — D tries to maximize its accuracy, G tries to minimize it.

## The Loss Function

The original GAN objective (Goodfellow et al. 2014):

```
min_G  max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

In practice we train with two separate BCE losses:

**Discriminator loss:**
```
L_D = BCE(D(x_real), 1) + BCE(D(G(z)), 0)
```

**Generator loss (non-saturating version):**
```
L_G = BCE(D(G(z)), 1)
```

The non-saturating version (G wants D to say "real") gives stronger gradients early in training.

## The Training Loop

```
for each batch:
    1. Sample noise z ~ N(0, I)
    2. Generate fake images: x_fake = G(z)

    3. Train D:
       - D(x_real) → should be 1
       - D(x_fake) → should be 0
       - Backprop through D only

    4. Train G:
       - D(G(z)) → G wants this to be 1
       - Backprop through G only (freeze D)
```

## Common Failure Modes

| Problem | Symptom | Cause |
|---------|---------|-------|
| **Mode Collapse** | G generates only one digit | G found one output that fools D |
| **Vanishing Gradients** | G loss stuck, no improvement | D is too strong, saturates BCE loss |
| **Training Instability** | Losses oscillate wildly | Learning rate too high or imbalanced |
| **Checkerboard artifacts** | Grid patterns in generated images | Upsampling artifacts (more in Phase 2) |

## Nash Equilibrium

The theoretical optimum is a **Nash Equilibrium** where:
- D outputs 0.5 for every image (can't tell real from fake)
- G produces samples indistinguishable from real data

In practice this is rarely reached perfectly — hence all the improvements in Phases 2–4.
