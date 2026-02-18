"""
Headless training script for MNIST multiclass classification with Quantum CNN.
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
    python -m src.headless.train_mnist
    python -m src.headless.train_mnist --output-dir runs/mnist_exp2 --seed 123
"""

import logging
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from ..qml.models.multiclass import BatchedGPUHybridQuantumMultiClassCNN
from ..qml.ansatz.dense import DenseQCNNAnsatz4
from ..training.trainers import MultiClassTrainer


CONFIG = {
    # Data
    "data_root": "src/data/MNIST",
    "image_size": 28,
    "limit_samples": None,
    # Model
    "num_classes": 10,
    "kernel_size": 3,
    "stride": 1,
    "encoding": "dense",
    "n_qubits": 4,
    "measurement": "z",
    "hidden_size": [64, 32],
    # Training
    "epochs": 20,
    "batch_size": 32,
    "lr": 0.002,
    "weight_decay": 1e-5,
    "max_grad_norm": 1.0,
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.5,
    "seed": 42,
    # Output
    "output_dir": "runs/mnist",
    "log_interval": 200,
    "save_every": 1,
}


def parse_cli_overrides():
    """Allow overriding output_dir, seed, and limit_samples from CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Quantum CNN on MNIST")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--limit-samples", type=int, default=None,
                        help="Limit dataset size for quick validation")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    args = parser.parse_args()

    config = CONFIG.copy()
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed
    if args.limit_samples is not None:
        config["limit_samples"] = args.limit_samples
    if args.epochs is not None:
        config["epochs"] = args.epochs
    return config


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(config):
    """Load and prepare MNIST train/test datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
    ])

    train_dataset_full = datasets.MNIST(
        root=config["data_root"], train=True, download=True, transform=transform
    )
    test_dataset_full = datasets.MNIST(
        root=config["data_root"], train=False, download=True, transform=transform
    )

    limit = config["limit_samples"]
    if limit is not None:
        train_dataset = Subset(
            train_dataset_full, range(min(limit, len(train_dataset_full)))
        )
        test_dataset = Subset(
            test_dataset_full, range(min(limit // 5, len(test_dataset_full)))
        )
    else:
        train_dataset = train_dataset_full
        test_dataset = test_dataset_full

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    return train_loader, test_loader, len(train_dataset), len(test_dataset)


def build_model(config, device):
    """Construct the quantum CNN model, selecting GPU-batched variant if CUDA."""

    model = BatchedGPUHybridQuantumMultiClassCNN(
        input_size=config["image_size"],
        num_classes=config["num_classes"],
        kernel_size=config["kernel_size"],
        stride=config["stride"],
        encoding=config["encoding"],
        ansatz=DenseQCNNAnsatz4(),
        n_qubits=config["n_qubits"],
        measurement=config["measurement"],
        hidden_size=config["hidden_size"],
    )
    return model.to(device)


def setup_logger(output_dir: str) -> logging.Logger:
    """Create a logger that writes to both a file and stdout."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f"train_mnist.{id(output_dir)}")
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
    train_loader, test_loader, n_train, n_test = load_data(config)

    # Model
    model = build_model(config, device)

    # Optimizer & loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # Learning rate scheduler: reduce LR by gamma every step_size epochs
    scheduler = StepLR(
        optimizer,
        step_size=config["scheduler_step_size"],
        gamma=config["scheduler_gamma"],
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
    trainer.save_config(config)
    logger.info(f"Train samples: {n_train}, Test samples: {n_test}")

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
