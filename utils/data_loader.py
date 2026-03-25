"""
utils/data_loader.py
Shared dataset utilities for all phases.
"""

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_mnist_loader(batch_size: int, image_size: int = 28, data_root: str = "../data") -> DataLoader:
    """
    Returns a DataLoader for the MNIST dataset.
    Images are normalized to [-1, 1] to match Tanh output of Generator.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # single channel
    ])

    dataset = torchvision.datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    return loader


def get_cifar10_loader(batch_size: int, image_size: int = 32, data_root: str = "../data") -> DataLoader:
    """
    Returns a DataLoader for the CIFAR-10 dataset.
    Images are normalized to [-1, 1] to match Tanh output of Generator.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 3 channels
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    return loader
