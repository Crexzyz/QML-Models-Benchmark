"""
Headless training script for MNIST multiclass classification with Quantum CNN.
Designed for queue-based HPC systems (SLURM, PBS, etc.).
All available examples, including the published test partition, are reshuffled
into this benchmark's reproducible stratified 80/10/10 split.

Outputs:
    <output_dir>/
        metrics.csv          - Per-epoch train/validation loss and accuracy
        training.log         - Detailed log with timestamps
        checkpoint_epoch_N.pt - Model checkpoint per epoch
        best_model.pt        - Best model by validation accuracy
        final_model.pt       - Final model state dict
        config.json          - Full training configuration for reproducibility
        split_manifest.json  - Fixed stratified 80/10/10 sample indices
        test_metrics.json    - Final held-out test metrics for best validation model

Usage:
    python -m src.headless.train_mnist
    python -m src.headless.train_mnist --output-dir runs/mnist_exp2 --seed 123
"""

import json
import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms

from ..qml.models.multiclass import MultiClassCNN, MultiClassQCNN
from ..training.trainers import MultiClassTrainer
from ..training.shared import (
    set_seed,
    seed_worker,
    build_ansatz,
    setup_logger,
    save_confusion_matrix,
    save_test_metrics,
    stratified_split_indices,
    limit_split_indices,
    split_class_counts,
)


CONFIG = {
    # Data
    "data_root": "src/data/MNIST",
    "image_size": 28,
    "limit_samples": None,
    "split_fractions": (0.8, 0.1, 0.1),
    # Model
    "num_classes": 10,
    "use_classical": False,
    "encoding": "dense",
    "ansatz": "dense",
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
    parser.add_argument(
        "--encoding",
        type=str,
        choices=["rx", "ry", "rz", "dense"],
        default=None,
        help="Quantum encoding strategy",
    )
    parser.add_argument(
        "--ansatz",
        type=str,
        choices=["rx", "ry", "rz", "dense"],
        default=None,
        help="Dense ansatz or single-axis ansatz for the QCNN",
    )
    parser.add_argument(
        "--measurement",
        type=str,
        choices=["x", "y", "z"],
        default=None,
        help="Measurement axis",
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
    if args.encoding is not None:
        config["encoding"] = args.encoding
    if args.ansatz is not None:
        config["ansatz"] = args.ansatz
    if args.measurement is not None:
        config["measurement"] = args.measurement

    # Auto-build output dir from config when not explicitly provided
    if args.output_dir is None:
        if config.get("use_classical", False):
            config["output_dir"] = "runs/mnist_classical"
        else:
            abbrev = {"dense": "d", "rx": "rx", "ry": "ry", "rz": "rz"}
            enc = abbrev.get(config["encoding"], config["encoding"])
            ans = abbrev.get(config["ansatz"], config["ansatz"])
            meas = config["measurement"]
            config["output_dir"] = f"runs/mnist_{enc}_{ans}_{meas}"

    return config


def load_data(config, use_cuda: bool):
    """Load MNIST and make a fixed stratified 80/10/10 split."""
    transform = transforms.Compose([
        # MultiClassQCNN expects 3 input channels.
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    ])

    train_source = datasets.MNIST(
        root=config["data_root"], train=True, download=True, transform=transform
    )
    test_source = datasets.MNIST(
        root=config["data_root"], train=False, download=True, transform=transform
    )
    full_dataset = ConcatDataset([train_source, test_source])
    targets = torch.cat([train_source.targets, test_source.targets]).numpy()
    train_indices, val_indices, test_indices = stratified_split_indices(
        targets, config["seed"], config["split_fractions"]
    )
    limit = config["limit_samples"]
    if limit is not None:
        val_test_limit = max(round(limit * 0.125), 1)
        train_indices = limit_split_indices(
            train_indices, targets, limit, config["seed"] + 1
        )
        val_indices = limit_split_indices(
            val_indices, targets, val_test_limit, config["seed"] + 2
        )
        test_indices = limit_split_indices(
            test_indices, targets, val_test_limit, config["seed"] + 3
        )
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    pin_memory = use_cuda
    persistent_workers = config["num_workers"] > 0
    train_generator = torch.Generator().manual_seed(config["seed"])
    val_generator = torch.Generator().manual_seed(config["seed"] + 1)
    test_generator = torch.Generator().manual_seed(config["seed"] + 2)
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        generator=val_generator,
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

    classes = [str(i) for i in range(config["num_classes"])]
    manifest = {
        "dataset": "MNIST",
        "seed": config["seed"],
        "fractions": config["split_fractions"],
        "total_samples": len(full_dataset),
        "train_indices": train_indices,
        "validation_indices": val_indices,
        "test_indices": test_indices,
        "class_counts": split_class_counts(
            targets,
            {
                "train": train_indices,
                "validation": val_indices,
                "test": test_indices,
            },
            classes,
        ),
        "is_smoke_test": limit is not None,
    }
    return (
        train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset),
        len(test_dataset), classes, manifest
    )


def build_model(config, device):
    """Construct the quantum CNN model, selecting GPU-batched variant if CUDA."""

    if config.get("use_classical", False):
        model = MultiClassCNN(num_classes=config["num_classes"])
        return model.to(device)

    model = MultiClassQCNN(
        num_classes=config["num_classes"],
        encoding=config["encoding"],
        ansatz=build_ansatz(config),
        readout_wires=[0, 1, 2, 3],
        measurement=config["measurement"],
        use_gpu=(device.type == "cuda"),
    )
    return model.to(device)


def main():
    config = parse_cli_overrides()
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    (
        train_loader, val_loader, test_loader, n_train, n_val, n_test, classes,
        split_manifest,
    ) = load_data(
        config, use_cuda=(device.type == "cuda")
    )

    # Model
    model = build_model(config, device)

    # Optimizer & loss
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # Learning rate scheduler monitors validation loss.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        min_lr=config["scheduler_min_lr"],
    )

    # Logger + trainer
    logger = setup_logger(config["output_dir"], logger_name="train_mnist")
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
    config["split_fractions"] = list(config["split_fractions"])
    trainer.save_config(config)
    with open(
        os.path.join(config["output_dir"], "split_manifest.json"),
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(split_manifest, manifest_file, indent=2)
    logger.info(
        f"Train samples: {n_train}, Validation samples: {n_val}, "
        f"Test samples: {n_test}"
    )

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
        validation_loader=val_loader,
        scheduler=scheduler,
    )

    best_model_path = os.path.join(config["output_dir"], "best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Validation-selected checkpoint not found: {best_model_path}"
        )
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded best validation model from {best_model_path}")

    test_metrics, confusion_matrix = trainer.evaluate(model, test_loader)
    cm_path = save_confusion_matrix(config["output_dir"], confusion_matrix, classes)
    save_test_metrics(
        config["output_dir"],
        test_metrics,
        {
            "selected_epoch": checkpoint["epoch"],
            "validation_accuracy": checkpoint["metrics"]["val_acc"],
            "selection_metric": "validation_accuracy",
            "seed": config["seed"],
            "split_fractions": config["split_fractions"],
            "dataset": "MNIST",
            "config": config,
        },
    )
    logger.info(
        "Final Test | "
        f"Loss={test_metrics['loss']:.4f}, Acc={test_metrics['acc']:.4f}"
    )
    logger.info(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
