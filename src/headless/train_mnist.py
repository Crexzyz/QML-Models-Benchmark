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
import os
import random
import sys
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms

from ..qml.models.multiclass import MultiClassCNN, MultiClassQCNN
from ..qml.ansatz.dense import DenseQCNNAnsatz4NoPool
from ..training.trainers import MultiClassTrainer


CONFIG = {
    # Data
    "data_root": "src/data/MNIST",
    "image_size": 28,
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
    "lr": 0.002,
    "weight_decay": 1e-5,
    "label_smoothing": 0.05,
    "max_grad_norm": 1.0,
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "scheduler_min_lr": 1e-5,
    "seed": 42,
    # Output
    "output_dir": "runs/mnist",
    "log_interval": 100,
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
    parser.add_argument(
        "--use-classical",
        action="store_true",
        help="Use MultiClassCNN instead of MultiClassQCNN",
    )
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
    if args.use_classical:
        config["use_classical"] = True
    return config


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id, base_seed):
    """Seed Python and NumPy RNGs in each DataLoader worker."""
    worker_seed = (base_seed + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_data(config, use_cuda: bool):
    """Load and prepare MNIST train/test datasets."""
    transform = transforms.Compose([
        # MultiClassQCNN expects 3 input channels.
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
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

    pin_memory = use_cuda
    persistent_workers = config["num_workers"] > 0
    train_generator = torch.Generator().manual_seed(config["seed"])
    test_generator = torch.Generator().manual_seed(config["seed"] + 1)
    worker_init_fn = partial(seed_worker, base_seed=config["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        generator=test_generator,
    )

    return train_loader, test_loader, len(train_dataset), len(test_dataset)


def build_model(config, device):
    """Construct the quantum CNN model, selecting GPU-batched variant if CUDA."""

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


def save_confusion_matrix(
    output_dir: str,
    confusion_matrix: np.ndarray,
    class_labels=None,
) -> str:
    """Save a labeled confusion matrix CSV and return its path."""
    import csv
    import os

    path = os.path.join(output_dir, "confusion_matrix_best.csv")
    if class_labels is None:
        class_labels = [str(i) for i in range(confusion_matrix.shape[0])]
    else:
        class_labels = [str(label) for label in class_labels]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", *class_labels])
        for idx, row in enumerate(confusion_matrix.astype(int)):
            writer.writerow([class_labels[idx], *row.tolist()])

    return path


def main():
    config = parse_cli_overrides()
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, test_loader, n_train, n_test = load_data(
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

    # Evaluate and export confusion matrix for the best checkpoint.
    import os

    best_model_path = os.path.join(config["output_dir"], "best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best model from {best_model_path}")
    else:
        logger.info("best_model.pt not found; using final model for evaluation")

    best_metrics, confusion_matrix = trainer.evaluate(model, test_loader)
    cm_path = save_confusion_matrix(config["output_dir"], confusion_matrix)
    logger.info(
        "Best Test | "
        f"Loss={best_metrics['loss']:.4f}, Acc={best_metrics['acc']:.4f}"
    )
    logger.info(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
