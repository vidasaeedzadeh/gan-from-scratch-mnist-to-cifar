# GAN From Scratch: MNIST to CIFAR-10

A step-by-step journey through Generative Adversarial Networks using **PyTorch**, progressing from a simple Vanilla GAN on MNIST to advanced variants on CIFAR-10.

## 🗺️ Roadmap

| Phase | Model | Dataset | Key Concepts |
|-------|-------|---------|--------------|
| 1 | [Vanilla GAN](./phase1_vanilla_gan/) | MNIST | MLP Generator & Discriminator, BCE loss, mode collapse |
| 2 | [DCGAN](./phase2_dcgan/) | MNIST → CIFAR-10 | Conv layers, BatchNorm, transposed convolutions |
| 3 | [WGAN-GP](./phase3_wgan_gp/) | CIFAR-10 | Wasserstein loss, gradient penalty, training stability |
| 4 | [Conditional GAN](./phase4_conditional/) | CIFAR-10 | Class conditioning, label embeddings |

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/vidasaeedzadeh/gan-from-scratch-mnist-to-cifar.git
cd gan-from-scratch-mnist-to-cifar

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Phase 1
cd phase1_vanilla_gan
python train.py
```

## 📁 Project Structure

```
gan-from-scratch-mnist-to-cifar/
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/                        # Theory notes per phase
│   ├── 01_theory_vanilla_gan.md
│   ├── 02_theory_dcgan.md
│   ├── 03_theory_wgan_gp.md
│   └── 04_theory_conditional.md
│
├── data/                        # Auto-downloaded datasets
│   └── README.md
│
├── utils/                       # Shared utilities
│   ├── data_loader.py
│   ├── visualize.py
│   └── metrics.py
│
├── phase1_vanilla_gan/
├── phase2_dcgan/
├── phase3_wgan_gp/
└── phase4_conditional/
```

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional but recommended)

## 📚 References

- [Original GAN Paper — Goodfellow et al. 2014](https://arxiv.org/abs/1406.2661)
- [DCGAN Paper — Radford et al. 2015](https://arxiv.org/abs/1511.06434)
- [WGAN Paper — Arjovsky et al. 2017](https://arxiv.org/abs/1701.07875)
- [WGAN-GP Paper — Gulrajani et al. 2017](https://arxiv.org/abs/1704.00028)
- [Conditional GAN — Mirza & Osindero 2014](https://arxiv.org/abs/1411.1784)
