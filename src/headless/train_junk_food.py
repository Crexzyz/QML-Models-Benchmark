"""
Headless training script for Junk Food binary classification with the shared
multiclass Quantum CNN backbone.
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
    python -m src.headless.train_junk_food
    python -m src.headless.train_junk_food --output-dir runs/experiment_2 --seed 123
"""

import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from ..datasets import JunkFoodBinaryDataset
from ..qml.models.multiclass import MultiClassCNN, MultiClassQCNN
from ..training.trainers import MultiClassTrainer
from ..training.shared import (
    build_ansatz,
    save_confusion_matrix,
    seed_worker,
    set_seed,
    setup_logger,
)


CONFIG = {
    # Data
    "train_data": "src/data/data_aug",
    "test_data": "src/data/data_noaug",
    "image_size": 64,
    "limit_samples": None,
    # Model
    "num_classes": 2,
    "use_classical": False,
    "encoding": "dense",
    "ansatz": "dense",
    "measurement": "z",
    # Training
    "epochs": 20,
    "batch_size": 16,
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
    "output_dir": "runs/junk_food",
    "log_interval": 20,
    "save_every": 1,
}


def parse_cli_overrides():
    """Allow overriding key run settings from the CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the shared Quantum CNN on Junk Food"
    )
    parser.add_argument("--train-data", type=str, default=None,
                        help="Junk food training directory")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Junk food evaluation directory")
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
    for key in (
        "train_data",
        "test_data",
        "seed",
        "limit_samples",
        "epochs",
        "image_size",
        "batch_size",
        "num_workers",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.use_classical:
        config["use_classical"] = True
    if args.encoding is not None:
        config["encoding"] = args.encoding
    if args.ansatz is not None:
        config["ansatz"] = args.ansatz
    if args.measurement is not None:
        config["measurement"] = args.measurement

    if args.output_dir is None:
        if config["use_classical"]:
            config["output_dir"] = "runs/junk_food_classical"
        else:
            abbrev = {"dense": "d", "rx": "rx", "ry": "ry", "rz": "rz"}
            enc = abbrev[config["encoding"]]
            ans = abbrev[config["ansatz"]]
            config["output_dir"] = (
                f"runs/junk_food_{enc}_{ans}_{config['measurement']}"
            )
    else:
        config["output_dir"] = args.output_dir

    return config


def load_data(config, use_cuda: bool):
    """Load and prepare train/test datasets."""
    transform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
        ]
    )

    train_dataset_full = JunkFoodBinaryDataset(
        config["train_data"], transform=transform
    )
    full_test_dataset = JunkFoodBinaryDataset(
        config["test_data"], transform=transform
    )

    generator = torch.Generator().manual_seed(config["seed"])
    limit = config["limit_samples"]
    if limit is not None:
        train_count = min(limit, len(train_dataset_full))
        test_count = min(max(limit // 5, 1), len(full_test_dataset))
        train_indices = torch.randperm(
            len(train_dataset_full), generator=generator
        )[:train_count]
        test_indices = torch.randperm(
            len(full_test_dataset), generator=generator
        )[:test_count]
        train_dataset = Subset(train_dataset_full, train_indices.tolist())
        test_dataset = Subset(full_test_dataset, test_indices.tolist())
    else:
        train_dataset = train_dataset_full
        target_test_size = min(
            int(len(train_dataset_full) * 0.25), len(full_test_dataset)
        )
        test_indices = torch.randperm(
            len(full_test_dataset), generator=generator
        )[:target_test_size]
        test_dataset = Subset(full_test_dataset, test_indices.tolist())

    pin_memory = use_cuda
    persistent_workers = config["num_workers"] > 0
    worker_init_fn = partial(seed_worker, base_seed=config["seed"])
    train_generator = torch.Generator().manual_seed(config["seed"])
    test_generator = torch.Generator().manual_seed(config["seed"] + 1)

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

    classes = ["no_food", "food"]
    return (
        train_loader,
        test_loader,
        len(train_dataset),
        len(test_dataset),
        classes,
    )


def build_model(config, device):
    """Construct the shared quantum CNN or its classical counterpart."""
    if config["use_classical"]:
        return MultiClassCNN(num_classes=config["num_classes"]).to(device)

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
    train_loader, test_loader, n_train, n_test, classes = load_data(
        config, use_cuda=(device.type == "cuda")
    )

    # Model
    model = build_model(config, device)

    # Optimizer & loss
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        min_lr=config["scheduler_min_lr"],
    )

    # Logger + trainer
    logger = setup_logger(config["output_dir"], logger_name="train_junk_food")
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
    config["classes"] = classes
    trainer.save_config(config)
    logger.info(f"Train samples: {n_train}, Test samples: {n_test}")
    logger.info(f"Classes ({len(classes)}): {classes}")

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
    best_model_path = os.path.join(config["output_dir"], "best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best model from {best_model_path}")
    else:
        logger.info("best_model.pt not found; using final model for evaluation")

    best_metrics, confusion_matrix = trainer.evaluate(model, test_loader)
    confusion_matrix_path = save_confusion_matrix(
        config["output_dir"], confusion_matrix, classes
    )
    logger.info(
        "Best Test | "
        f"Loss={best_metrics['loss']:.4f}, Acc={best_metrics['acc']:.4f}"
    )
    logger.info(f"Confusion matrix saved to {confusion_matrix_path}")


if __name__ == "__main__":
    main()
