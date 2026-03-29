# GAN-CelebA

A deep learning project for learning and implementing a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic human face images using the CelebA dataset.

---

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results](#results)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

---

## Motivation

Generative Adversarial Networks (GANs) are one of the most exciting and challenging areas of deep learning. This project was built as a hands-on learning exercise to understand:

- The theory and mechanics behind GANs
- How a Generator and Discriminator are trained adversarially
- The practical challenges of GAN training such as mode collapse, instability, and hyperparameter sensitivity
- How to implement and debug a DCGAN from scratch using PyTorch

The goal is not just to produce good results, but to deeply understand each component of the GAN training pipeline step by step.

---

## Dataset

**CelebA (Large-scale CelebFaces Attributes Dataset)**

- **Source:** [Kaggle - jessicali9530/celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- **Size:** 202,599 celebrity face images
- **Original resolution:** 218 x 178 pixels
- **Preprocessed resolution:** 64 x 64 pixels (resized and center-cropped)
- **Normalization:** Pixel values normalized to the range `[-1, 1]` using mean and std of `0.5` per channel

### Preprocessing Pipeline

```python
transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

---

## Model Architecture

The architecture is based on the **DCGAN paper** by Radford et al. (2015), adapted for 64x64 RGB face generation.

### Generator

The Generator takes a random noise vector `z` of size 100 (latent dimension) and progressively upsamples it into a 64x64 RGB image using transposed convolutions.

| Layer | Output Shape | Details |
|---|---|---|
| Input (noise) | `[batch, 100, 1, 1]` | Random normal vector |
| ConvTranspose2d | `[batch, 1024, 4, 4]` | kernel=4, stride=1, padding=0 |
| ConvTranspose2d | `[batch, 512, 8, 8]` | kernel=4, stride=2, padding=1 |
| ConvTranspose2d | `[batch, 256, 16, 16]` | kernel=4, stride=2, padding=1 |
| ConvTranspose2d | `[batch, 128, 32, 32]` | kernel=4, stride=2, padding=1 |
| ConvTranspose2d | `[batch, 3, 64, 64]` | kernel=4, stride=2, padding=1 |
| Output | `[batch, 3, 64, 64]` | Tanh activation |

- **Hidden activations:** ReLU
- **Final activation:** Tanh (outputs values in `[-1, 1]`)
- **Normalization:** BatchNorm2d after every layer except the last
- **Total parameters:** ~13.7 million

### Discriminator

The Discriminator takes a 64x64 RGB image (real or fake) and outputs a single value indicating the probability of the image being real.

| Layer | Output Shape | Details |
|---|---|---|
| Input | `[batch, 3, 64, 64]` | Real or fake image |
| Conv2d | `[batch, 128, 32, 32]` | kernel=4, stride=2, padding=1 |
| Conv2d | `[batch, 256, 16, 16]` | kernel=4, stride=2, padding=1 |
| Conv2d | `[batch, 512, 8, 8]` | kernel=4, stride=2, padding=1 |
| Conv2d | `[batch, 1024, 4, 4]` | kernel=4, stride=2, padding=1 |
| Conv2d | `[batch, 1, 1, 1]` | kernel=4, stride=1, padding=0 |
| Output | `[batch, 1]` | Single probability score |

- **Hidden activations:** LeakyReLU (slope=0.2)
- **Normalization:** BatchNorm2d after every layer except the first
- **Total parameters:** ~11 million

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Generator learning rate | 0.0002 |
| Discriminator learning rate | 0.0001 |
| Beta1 | 0.5 |
| Beta2 | 0.999 |
| Batch size | 128 |
| Epochs | 40 |
| Latent dimension | 100 |
| Loss function | BCEWithLogitsLoss |
| Real labels (smoothed) | 0.9 |
| Fake labels (smoothed) | 0.1 |

### Key Training Decisions

**Label Smoothing:** Instead of hard labels of `0` and `1`, we use `0.9` for real and `0.1` for fake. This prevents the Discriminator from becoming overconfident and helps stabilize training.

**Weight Initialization:** All Conv and ConvTranspose layers are initialized with a normal distribution (mean=0, std=0.02). BatchNorm layers are initialized with weight=1 and bias=0, following the DCGAN paper.

**`detach()` in Discriminator step:** When training the Discriminator on fake images, `fake_batch.detach()` is used to prevent gradients from flowing back into the Generator, ensuring each network is updated independently.

### Training Loop Overview

Each batch consists of two steps:

1. **Train the Discriminator:**
   - Feed real images → compute loss against label `0.9`
   - Feed fake images (detached) → compute loss against label `0.1`
   - Total loss = real loss + fake loss → update Discriminator weights

2. **Train the Generator:**
   - Feed the same fake images through the updated Discriminator
   - Compute loss against label `0.9` (Generator wants Discriminator to think fakes are real)
   - Update Generator weights

---

## Results

### Metrics Tracked Per Epoch

- `Loss_G` — Generator loss
- `Loss_D` — Discriminator loss (real + fake)
- `D_real` — Average Discriminator output on real images (target: ~0.5)
- `D_fake` — Average Discriminator output on fake images (target: ~0.5)

### Healthy Training Signs

| Metric | Early Training | Healthy Training |
|---|---|---|
| `D_real` | ~0.7-0.9 | ~0.5-0.7 |
| `D_fake` | ~0.1-0.3 | ~0.3-0.5 |
| `Loss_D` | ~1.0-1.5 | stable ~1.0 |
| `Loss_G` | ~2.0-3.0 | decreasing |

### Generated Images Per Epoch

A fixed noise vector is used at the end of every epoch to generate the same set of 64 images. This allows us to visually track how the Generator improves over time — from random noise in early epochs to increasingly realistic faces in later epochs.

> Generated image samples will be added here after training completes.

### Loss Curves

Training loss for both the Generator (`Loss_G`) and Discriminator (`Loss_D`) are tracked across all epochs and plotted to monitor training stability. A healthy training run shows `Loss_G` gradually decreasing while `Loss_D` remains relatively stable.

> Loss curve plot will be added here after training completes.

### D_real and D_fake Plots

`D_real` and `D_fake` are plotted across epochs to visualize the adversarial balance between the two networks. The ideal outcome is both values converging toward `0.5`, meaning the Discriminator can no longer reliably distinguish real from fake images.

> D_real / D_fake plot will be added here after training completes.

---

## How to Run

### On Kaggle (Recommended)

1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook
2. Add the CelebA dataset:
   - Click **+ Add Input** in the right panel
   - Search for `jessicali9530/celeba-dataset` and add it
3. Enable GPU:
   - Right panel → **Session Options** → **Accelerator → GPU T4 x2**
4. Clone this repository in a notebook cell:
   ```bash
   !git clone https://github.com/your-username/GAN-CelebA.git
   ```
5. Run the notebook cells in order

### Verify GPU is Available

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)  # Should print: cuda
```

### Dataset Path on Kaggle

```python
data_folder = '/kaggle/input/datasets/jessicali9530/celeba-dataset/img_align_celeba/img_align_celeba/'
```

---

## Project Structure

```
GAN-CelebA/
│
├── notebooks/              # Kaggle Jupyter notebooks
│   └── dcgan_celeba.ipynb
│
├── src/                    # Reusable Python modules
│   ├── dataset.py          # CelebADataset class
│   ├── generator.py        # Generator model
│   ├── discriminator.py    # Discriminator model
│   └── train.py            # Training loop
│
├── outputs/                # Generated images and model checkpoints
│   ├── images/             # Generated face samples per epoch
│   └── checkpoints/        # Saved model weights
│
├── data/                   # Data scripts (not the dataset itself)
│
└── README.md
```

---

## Dependencies

```
torch
torchvision
torchsummary
numpy
matplotlib
Pillow
pandas
```

Install with:
```bash
pip install torch torchvision torchsummary numpy matplotlib Pillow pandas
```

---

## References

- Radford, A., Metz, L., & Chintala, S. (2015). [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). [Deep Learning Face Attributes in the Wild](https://arxiv.org/abs/1411.7766) (CelebA dataset)
- Goodfellow, I., et al. (2014). [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)