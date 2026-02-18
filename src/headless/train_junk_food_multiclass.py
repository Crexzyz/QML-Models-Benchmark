"""
Headless training script for Junk Food multi-label classification with Quantum CNN.
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
    python -m src.headless.train_junk_food_multiclass
"""

import logging
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from ..datasets import JunkFoodMulticlassDataset
from ..qml.models.multiclass import BatchedGPUHybridQuantumMultiClassCNN
from ..qml.ansatz.dense import DenseQCNNAnsatz4
from ..training.trainers import MultiLabelTrainer


CONFIG = {
    # Data
    "train_data": "src/data/data_aug",
    "test_data": "src/data/data_noaug",
    "image_size": 128,
    # Model
    "kernel_size": 3,
    "stride": 1,
    "pool_size": 12,
    "encoding": "dense",
    "n_qubits": 4,
    "measurement": "x",
    "hidden_size": [128, 64, 32],
    # Training
    "epochs": 15,
    "batch_size": 32,
    "lr": 0.005,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "seed": 42,
    # Output
    "output_dir": "runs/junk_food_multiclass",
    "log_interval": 10,
    "save_every": 1,
}


def parse_cli_overrides():
    """Allow overriding output_dir and seed from CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Quantum CNN on Junk Food (Multiclass)"
    )
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    args = parser.parse_args()

    config = CONFIG.copy()
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(config):
    """Load and prepare train/test datasets."""
    transform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
        ]
    )

    train_dataset = JunkFoodMulticlassDataset(
        config["train_data"], transform=transform
    )
    full_test_dataset = JunkFoodMulticlassDataset(
        config["test_data"], transform=transform
    )

    # Create ~80/20 split based on training size
    target_test_size = int(len(train_dataset) * 0.25)
    indices = list(range(len(full_test_dataset)))
    random.seed(config["seed"])
    random.shuffle(indices)
    test_dataset = Subset(full_test_dataset, indices[:target_test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    num_classes = train_dataset.num_classes
    classes = train_dataset.classes

    return (
        train_loader, test_loader, len(train_dataset), len(test_dataset),
        num_classes, classes
    )


def build_model(config, num_classes, device):
    """Construct the quantum CNN model for multiclass classification."""
    model = BatchedGPUHybridQuantumMultiClassCNN(
        num_classes=num_classes,
        input_size=config["image_size"],
        kernel_size=config["kernel_size"],
        stride=config["stride"],
        pool_size=config["pool_size"],
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
    logger = logging.getLogger(f"train_junk_food_multiclass.{id(output_dir)}")
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
        config
    )

    # Model
    model = build_model(config, num_classes, device)

    # Optimizer & loss
    # BCEWithLogitsLoss for multi-label (multi-hot) classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # Logger + trainer
    logger = setup_logger(config["output_dir"])
    trainer = MultiLabelTrainer(
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
    )


if __name__ == "__main__":
    main()
