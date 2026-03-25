# Data

Datasets are **automatically downloaded** by `torchvision` when you run any training script for the first time. You do not need to manually download anything.

## Datasets Used

| Phase | Dataset | Size | Auto-downloaded |
|-------|---------|------|-----------------|
| 1 & 2 (start) | MNIST | ~11 MB | ✅ Yes |
| 2 (end) & 3 & 4 | CIFAR-10 | ~163 MB | ✅ Yes |

## Storage

Downloaded data is saved to this `data/` folder but is **excluded from git** via `.gitignore` to avoid committing large binary files.
