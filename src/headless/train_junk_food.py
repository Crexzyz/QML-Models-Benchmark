"""
Headless training script for Junk Food binary classification with the shared
multiclass Quantum CNN backbone.
Designed for queue-based HPC systems (SLURM, PBS, etc.).

Outputs:
    <output_dir>/
        metrics.csv          - Per-epoch metrics (legacy test_* columns)
        training.log         - Detailed log with timestamps
        checkpoint_epoch_N.pt - Model checkpoint per epoch
        best_model.pt        - Best model by validation accuracy
        final_model.pt       - Final model state dict
        config.json          - Full training configuration for reproducibility
        validation_split.json - Source-level train/validation split summary

Usage:
    python -m src.headless.train_junk_food
    python -m src.headless.train_junk_food --output-dir runs/experiment_2 --seed 123
"""

import json
import os
from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, Subset
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
    "validation_data": "src/data/data_noaug",
    "validation_fraction": 0.20,
    "split_seed": 42,
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
    parser.add_argument("--validation-data", "--test-data", dest="validation_data",
                        type=str, default=None,
                        help="Junk food validation directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Override the fixed source-level split seed")
    parser.add_argument("--limit-samples", type=int, default=None,
                        help="Limit training examples and validation proportionally")
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
        "validation_data",
        "split_seed",
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


def _source_key(file_name: str) -> str:
    """Return the Roboflow source filename shared by augmented variants."""
    source_key, separator, _ = file_name.partition(".rf.")
    if not separator or not source_key:
        raise ValueError(
            f"Expected a Roboflow filename containing '.rf.': {file_name}"
        )
    return source_key


def _make_source_level_split(augmented_dataset, original_dataset, config):
    """Keep each original image and all its augmented variants in one split."""
    validation_fraction = config["validation_fraction"]
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    original_indices_by_source = defaultdict(list)
    original_labels_by_source = {}
    for index, image in enumerate(original_dataset.images):
        source = _source_key(image["file_name"])
        label = int(image["has_food"])
        original_indices_by_source[source].append(index)
        if source in original_labels_by_source:
            if original_labels_by_source[source] != label:
                raise ValueError(
                    f"Original source {source} has conflicting labels"
                )
        else:
            original_labels_by_source[source] = label

    augmented_indices_by_source = defaultdict(list)
    augmented_labels_by_source = defaultdict(set)
    for index, image in enumerate(augmented_dataset.images):
        source = _source_key(image["file_name"])
        augmented_indices_by_source[source].append(index)
        augmented_labels_by_source[source].add(int(image["has_food"]))

    original_sources = set(original_indices_by_source)
    augmented_sources = set(augmented_indices_by_source)
    missing_originals = augmented_sources - original_sources
    if missing_originals:
        examples = sorted(missing_originals)[:5]
        raise ValueError(
            "Augmented images do not map to original images; "
            f"example source keys: {examples}"
        )

    sources_by_label = defaultdict(list)
    for source, label in original_labels_by_source.items():
        sources_by_label[label].append(source)

    target_validation_sources = round(
        len(original_sources) * validation_fraction
    )
    quotas = {
        label: len(sources) * validation_fraction
        for label, sources in sources_by_label.items()
    }
    validation_counts = {
        label: int(quota) for label, quota in quotas.items()
    }
    remaining = target_validation_sources - sum(validation_counts.values())
    labels_by_remainder = sorted(
        quotas,
        key=lambda label: (
            -(quotas[label] - validation_counts[label]),
            label,
        ),
    )
    for label in labels_by_remainder[:remaining]:
        validation_counts[label] += 1

    split_generator = torch.Generator().manual_seed(config["split_seed"])
    validation_sources = set()
    for label in sorted(sources_by_label):
        sources = sources_by_label[label]
        order = torch.randperm(
            len(sources), generator=split_generator
        ).tolist()
        validation_sources.update(
            sources[index]
            for index in order[:validation_counts[label]]
        )

    training_sources = original_sources - validation_sources
    train_augmented_indices = [
        index
        for source, indices in augmented_indices_by_source.items()
        if source in training_sources
        for index in indices
    ]
    train_original_only_indices = [
        index
        for source, indices in original_indices_by_source.items()
        if source in training_sources and source not in augmented_sources
        for index in indices
    ]
    validation_indices = [
        index
        for source, indices in original_indices_by_source.items()
        if source in validation_sources
        for index in indices
    ]

    selected_training_sources = {
        _source_key(augmented_dataset.images[index]["file_name"])
        for index in train_augmented_indices
    }
    if selected_training_sources & validation_sources:
        raise RuntimeError("A validation source also has augmented training data")
    if not validation_indices or not (
        train_augmented_indices or train_original_only_indices
    ):
        raise ValueError("The source-level split produced an empty partition")

    training_dataset = ConcatDataset(
        [
            Subset(augmented_dataset, train_augmented_indices),
            Subset(original_dataset, train_original_only_indices),
        ]
    )
    validation_dataset = Subset(original_dataset, validation_indices)
    split_summary = {
        "source_key_rule": "filename prefix before '.rf.'",
        "split_seed": config["split_seed"],
        "validation_fraction": validation_fraction,
        "training_source_count": len(training_sources),
        "validation_source_count": len(validation_sources),
        "training_augmented_sample_count": len(train_augmented_indices),
        "training_original_only_sample_count": len(train_original_only_indices),
        "validation_original_sample_count": len(validation_indices),
        "training_sources_by_class": {
            str(label): len(sources_by_label[label]) - validation_counts[label]
            for label in sorted(sources_by_label)
        },
        "validation_sources_by_class": {
            str(label): validation_counts[label]
            for label in sorted(sources_by_label)
        },
        "augmented_source_groups_with_mixed_labels": sorted(
            source
            for source, labels in augmented_labels_by_source.items()
            if len(labels) > 1
        ),
        "validation_sources": sorted(validation_sources),
    }
    return training_dataset, validation_dataset, split_summary


def load_data(config, use_cuda: bool):
    """Load training variants and an isolated, source-level validation split."""
    transform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
        ]
    )

    augmented_dataset = JunkFoodBinaryDataset(
        config["train_data"], transform=transform
    )
    original_dataset = JunkFoodBinaryDataset(
        config["validation_data"], transform=transform
    )

    training_dataset, validation_dataset, split_summary = (
        _make_source_level_split(
            augmented_dataset,
            original_dataset,
            config,
        )
    )

    limit = config["limit_samples"]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit_samples must be a positive integer")
        subset_generator = torch.Generator().manual_seed(
            config["split_seed"] + 1
        )
        train_count = min(limit, len(training_dataset))
        validation_count = min(
            max(
                round(
                    train_count
                    * config["validation_fraction"]
                    / (1 - config["validation_fraction"])
                ),
                1,
            ),
            len(validation_dataset),
        )
        training_dataset = Subset(
            training_dataset,
            torch.randperm(
                len(training_dataset), generator=subset_generator
            )[:train_count].tolist(),
        )
        validation_dataset = Subset(
            validation_dataset,
            torch.randperm(
                len(validation_dataset), generator=subset_generator
            )[:validation_count].tolist(),
        )

    pin_memory = use_cuda
    persistent_workers = config["num_workers"] > 0
    worker_init_fn = partial(seed_worker, base_seed=config["seed"])
    train_generator = torch.Generator().manual_seed(config["seed"])
    validation_generator = torch.Generator().manual_seed(config["seed"] + 1)

    train_loader = DataLoader(
        training_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        generator=validation_generator,
    )

    classes = ["no_food", "food"]
    return (
        train_loader,
        validation_loader,
        len(training_dataset),
        len(validation_dataset),
        classes,
        split_summary,
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
    (
        train_loader,
        validation_loader,
        n_train,
        n_validation,
        classes,
        split_summary,
    ) = load_data(config, use_cuda=(device.type == "cuda"))

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
    with open(
        os.path.join(config["output_dir"], "validation_split.json"),
        "w",
        encoding="utf-8",
    ) as split_file:
        json.dump(split_summary, split_file, indent=2)
    logger.info(
        f"Train samples: {n_train}, Test samples: {n_validation} "
        "(validation split)"
    )
    logger.info(
        "Source-level split: "
        f"{split_summary['training_source_count']} train sources, "
        f"{split_summary['validation_source_count']} validation sources"
    )
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
        test_loader=validation_loader,
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

    best_metrics, confusion_matrix = trainer.evaluate(
        model, validation_loader
    )
    confusion_matrix_path = save_confusion_matrix(
        config["output_dir"], confusion_matrix, classes
    )
    logger.info(
        "Best Test (validation split) | "
        f"Loss={best_metrics['loss']:.4f}, Acc={best_metrics['acc']:.4f}"
    )
    logger.info(f"Confusion matrix saved to {confusion_matrix_path}")


if __name__ == "__main__":
    main()
