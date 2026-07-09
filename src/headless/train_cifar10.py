"""
Headless training script for CIFAR-10 multiclass classification with Quantum CNN.
Designed for queue-based HPC systems (SLURM, PBS, etc.).

Outputs:
    <output_dir>/
        metrics.csv          - Per-epoch train/test loss and accuracy
        training.log         - Detailed log with timestamps
        checkpoint_epoch_N.pt - Model checkpoint per epoch
        best_model.pt        - Best model by test accuracy
        final_model.pt       - Final model state dict
        config.json          - Full training configuration for reproducibility

Usage:
    python -m src.headless.train_cifar10
    python -m src.headless.train_cifar10 --output-dir runs/cifar10_exp2 --seed 123
"""

import logging
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ..qml.ansatz.dense import DenseQCNNAnsatz4NoPool
from ..qml.models.multiclass import MultiClassCNN, MultiClassQCNN
from ..training.trainers import MultiClassTrainer


CONFIG = {
    # Data
    "data_root": "src/data/cifar10",
    "image_size": 32,
    "limit_samples": None,
    # Model
    "num_classes": 10,
    "use_classical": False,
    "encoding": "dense",
    "measurement": "z",
    # Training
    "epochs": 20,
    "batch_size": 32,
    "num_workers": 2,
    "lr": 0.0015,
    "weight_decay": 1e-5,
    "label_smoothing": 0.05,
    "max_grad_norm": 1.0,
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "scheduler_min_lr": 1e-5,
    "seed": 42,
    # Output
    "output_dir": "runs/cifar10",
    "log_interval": 20,
    "save_every": 1,
}


def parse_cli_overrides():
    """Allow overriding key run settings from the CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Quantum CNN on CIFAR-10")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override dataset root directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--limit-samples", type=int, default=None,
                        help="Limit dataset size for quick validation")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Override square resize size")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override DataLoader worker count")
    parser.add_argument(
        "--use-classical",
        action="store_true",
        help="Use MultiClassCNN instead of MultiClassQCNN",
    )
    args = parser.parse_args()

    config = CONFIG.copy()
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed
    if args.limit_samples is not None:
        config["limit_samples"] = args.limit_samples
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.image_size is not None:
        config["image_size"] = args.image_size
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.use_classical:
        config["use_classical"] = True
    return config


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(config, use_cuda: bool):
    """Load and prepare CIFAR-10 train/test datasets."""
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.RandomResizedCrop(
            config["image_size"], scale=(0.7, 1.0), ratio=(0.8, 1.25)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(config["image_size"], padding=4),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
        # CIFAR-style cutout-like regularization.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    train_dataset_full = datasets.CIFAR10(
        root=config["data_root"], train=True, download=True, transform=train_transform
    )
    test_dataset_full = datasets.CIFAR10(
        root=config["data_root"], train=False, download=True, transform=test_transform
    )

    limit = config["limit_samples"]
    generator = torch.Generator().manual_seed(config["seed"])
    if limit is not None:
        train_count = min(limit, len(train_dataset_full))
        test_count = min(max(limit // 5, 1), len(test_dataset_full))
        train_indices = torch.randperm(
            len(train_dataset_full), generator=generator
        )[:train_count]
        test_indices = torch.randperm(
            len(test_dataset_full), generator=generator
        )[:test_count]
        train_dataset = Subset(train_dataset_full, train_indices.tolist())
        test_dataset = Subset(test_dataset_full, test_indices.tolist())
    else:
        train_dataset = train_dataset_full
        test_dataset = test_dataset_full

    pin_memory = use_cuda
    persistent_workers = config["num_workers"] > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    classes = train_dataset_full.classes
    return (
        train_loader,
        test_loader,
        len(train_dataset),
        len(test_dataset),
        len(classes),
        classes,
    )


def build_model(config, device):
    """Construct the quantum CNN model."""

    if config.get("use_classical", False):
        model = MultiClassCNN(num_classes=config["num_classes"])
        return model.to(device)

    model = MultiClassQCNN(
        num_classes=config["num_classes"],
        encoding=config["encoding"],
        ansatz=DenseQCNNAnsatz4NoPool(),
        readout_wires=[0, 1, 2, 3],
        measurement=config["measurement"],
        use_gpu=(device.type == "cuda"),
    )
    return model.to(device)


def setup_logger(output_dir: str) -> logging.Logger:
    """Create a logger that writes to both a file and stdout."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f"train_cifar10.{id(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(os.path.join(output_dir, "training.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def main():
    config = parse_cli_overrides()
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, test_loader, n_train, n_test, num_classes, classes = load_data(
        config, use_cuda=(device.type == "cuda")
    )

    # Model
    model = build_model(config, device)

    # Optimizer & loss
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # Learning rate scheduler: reduce LR when validation/test loss plateaus.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        min_lr=config["scheduler_min_lr"],
    )

    # Logger + trainer
    logger = setup_logger(config["output_dir"])
    trainer = MultiClassTrainer(
        criterion=criterion,
        device=device,
        max_grad_norm=config["max_grad_norm"],
        log_interval=config["log_interval"],
        logger=logger,
        output_dir=config["output_dir"],
        save_every=config["save_every"],
    )

    # Save config & log setup info
    config["device"] = str(device)
    config["num_classes"] = num_classes
    config["classes"] = classes
    trainer.save_config(config)
    logger.info(f"Train samples: {n_train}, Test samples: {n_test}")
    if num_classes > 5:
        logger.info(f"Classes ({num_classes}): {classes[:5]}...")
    else:
        logger.info(f"Classes ({num_classes}): {classes}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
    )

    # Train
    trainer.train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=config["epochs"],
        test_loader=test_loader,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
