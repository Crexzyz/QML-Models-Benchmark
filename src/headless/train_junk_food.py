"""
Headless training script for Junk Food binary classification with the shared
multiclass Quantum CNN backbone.
Designed for queue-based HPC systems (SLURM, PBS, etc.).

Outputs:
    <output_dir>/
        metrics.csv          - Per-epoch train/validation metrics
        training.log         - Detailed log with timestamps
        checkpoint_epoch_N.pt - Model checkpoint per epoch
        best_model.pt        - Best model by validation accuracy
        final_model.pt       - Final model state dict
        config.json          - Full training configuration for reproducibility
        split_manifest.json  - Source-level stratified 80/10/10 split
        test_metrics.json    - Final held-out test metrics for best validation model

Usage:
    python -m src.headless.train_junk_food
    python -m src.headless.train_junk_food --output-dir runs/experiment_2 --seed 123
"""

import json
import os
from collections import defaultdict
from functools import partial

import numpy as np
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
    stratified_split_indices,
    limit_split_indices,
    save_test_metrics,
)


CONFIG = {
    # Data
    "train_data": "src/data/data_aug",
    "validation_data": "src/data/data_noaug",
    "split_fractions": (0.8, 0.1, 0.1),
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
                        help="Directory containing the original labeled images")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--limit-samples", type=int, default=None,
                        help="Limit each split by source count for smoke tests")
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
    """Stratify original sources; keep all augmented variants in training only."""
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

    ordered_sources = sorted(original_sources)
    source_targets = np.asarray(
        [original_labels_by_source[source] for source in ordered_sources]
    )
    train_source_indices, validation_source_indices, test_source_indices = (
        stratified_split_indices(
            source_targets, config["seed"], config["split_fractions"]
        )
    )
    source_splits = [
        {ordered_sources[index] for index in indices}
        for indices in (
            train_source_indices, validation_source_indices, test_source_indices
        )
    ]
    limit = config["limit_samples"]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit_samples must be a positive integer")
        split_caps = (limit, max(round(limit * 0.125), 1),
                      max(round(limit * 0.125), 1))
        limited_source_splits = []
        for split_index, (source_indices, cap) in enumerate(
            zip(
                (train_source_indices, validation_source_indices, test_source_indices),
                split_caps,
            )
        ):
            chosen_indices = limit_split_indices(
                source_indices,
                source_targets,
                cap,
                config["seed"] + split_index + 1,
            )
            limited_source_splits.append(
                {ordered_sources[index] for index in chosen_indices}
            )
        source_splits = limited_source_splits

    training_sources, validation_sources, test_sources = source_splits
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
    test_indices = [
        index
        for source, indices in original_indices_by_source.items()
        if source in test_sources
        for index in indices
    ]

    selected_training_sources = {
        _source_key(augmented_dataset.images[index]["file_name"])
        for index in train_augmented_indices
    }
    if selected_training_sources & (validation_sources | test_sources):
        raise RuntimeError("Validation/test source has augmented training data")
    if not validation_indices or not test_indices or not (
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
    test_dataset = Subset(original_dataset, test_indices)
    split_summary = {
        "source_key_rule": "filename prefix before '.rf.'",
        "seed": config["seed"],
        "fractions": config["split_fractions"],
        "is_smoke_test": config["limit_samples"] is not None,
        "total_source_count": len(original_sources),
        "training_source_count": len(training_sources),
        "validation_source_count": len(validation_sources),
        "test_source_count": len(test_sources),
        "training_augmented_sample_count": len(train_augmented_indices),
        "training_original_only_sample_count": len(train_original_only_indices),
        "validation_original_sample_count": len(validation_indices),
        "test_original_sample_count": len(test_indices),
        "source_class_counts": {
            phase: {
                str(label): sum(
                    original_labels_by_source[source] == label
                    for source in sources
                )
                for label in sorted(set(original_labels_by_source.values()))
            }
            for phase, sources in zip(
                ("train", "validation", "test"),
                (training_sources, validation_sources, test_sources),
            )
        },
        "augmented_source_groups_with_mixed_labels": sorted(
            source
            for source, labels in augmented_labels_by_source.items()
            if len(labels) > 1
        ),
        "training_sources": sorted(training_sources),
        "validation_sources": sorted(validation_sources),
        "test_sources": sorted(test_sources),
    }
    return training_dataset, validation_dataset, test_dataset, split_summary


def load_data(config, use_cuda: bool):
    """Load Junk Food with source-isolated 80/10/10 binary splits."""
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

    training_dataset, validation_dataset, test_dataset, split_summary = (
        _make_source_level_split(
            augmented_dataset,
            original_dataset,
            config,
        )
    )

    pin_memory = use_cuda
    persistent_workers = config["num_workers"] > 0
    worker_init_fn = partial(seed_worker, base_seed=config["seed"])
    train_generator = torch.Generator().manual_seed(config["seed"])
    validation_generator = torch.Generator().manual_seed(config["seed"] + 1)
    test_generator = torch.Generator().manual_seed(config["seed"] + 2)

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
        validation_loader,
        test_loader,
        len(training_dataset),
        len(validation_dataset),
        len(test_dataset),
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
        test_loader,
        n_train,
        n_validation,
        n_test,
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
    config["split_fractions"] = list(config["split_fractions"])
    trainer.save_config(config)
    with open(
        os.path.join(config["output_dir"], "split_manifest.json"),
        "w",
        encoding="utf-8",
    ) as split_file:
        json.dump(split_summary, split_file, indent=2)
    logger.info(
        f"Train samples: {n_train}, Validation samples: {n_validation}, "
        f"Test samples: {n_test}"
    )
    logger.info(
        "Source-level split: "
        f"{split_summary['training_source_count']} train sources, "
        f"{split_summary['validation_source_count']} validation sources, "
        f"{split_summary['test_source_count']} test sources"
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
        validation_loader=validation_loader,
        scheduler=scheduler,
    )

    # Evaluate and export confusion matrix for the best checkpoint.
    best_model_path = os.path.join(config["output_dir"], "best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Validation-selected checkpoint not found: {best_model_path}"
        )
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded best validation model from {best_model_path}")

    test_metrics, confusion_matrix = trainer.evaluate(model, test_loader)
    confusion_matrix_path = save_confusion_matrix(
        config["output_dir"], confusion_matrix, classes
    )
    save_test_metrics(
        config["output_dir"],
        test_metrics,
        {
            "selected_epoch": checkpoint["epoch"],
            "validation_accuracy": checkpoint["metrics"]["val_acc"],
            "selection_metric": "validation_accuracy",
            "seed": config["seed"],
            "split_fractions": config["split_fractions"],
            "dataset": "Junk Food (binary)",
            "config": config,
        },
    )
    logger.info(
        "Final Test | "
        f"Loss={test_metrics['loss']:.4f}, Acc={test_metrics['acc']:.4f}"
    )
    logger.info(f"Confusion matrix saved to {confusion_matrix_path}")


if __name__ == "__main__":
    main()
