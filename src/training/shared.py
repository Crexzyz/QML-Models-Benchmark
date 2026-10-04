"""
Shared utility functions for headless training scripts.
"""

import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch

from ..qml.ansatz.dense import DenseQCNNAnsatz4NoPool, SingleAxisQCNNAnsatz4NoPool


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


def stratified_split_indices(
    targets,
    seed: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
):
    """Return deterministic, disjoint train/validation/test indices by class."""
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must contain three values summing to 1")
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("Each split fraction must be greater than zero")

    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    splits = ([], [], [])
    for label in np.unique(targets):
        indices = np.flatnonzero(targets == label)
        rng.shuffle(indices)
        count = len(indices)
        train_end = int(count * fractions[0])
        validation_end = train_end + int(count * fractions[1])
        if min(train_end, validation_end - train_end, count - validation_end) < 1:
            raise ValueError(
                f"Class {label!r} needs at least three samples for 80/10/10 splits"
            )
        splits[0].extend(indices[:train_end].tolist())
        splits[1].extend(indices[train_end:validation_end].tolist())
        splits[2].extend(indices[validation_end:].tolist())

    return tuple(split for split in splits)


def limit_split_indices(indices, targets, limit: int | None, seed: int):
    """Deterministically cap a split while approximately preserving class ratios."""
    if limit is None or len(indices) <= limit:
        return list(indices)
    if limit < 1:
        raise ValueError("limit_samples must be a positive integer")

    indices = np.asarray(indices)
    split_targets = np.asarray(targets)[indices]
    classes, counts = np.unique(split_targets, return_counts=True)
    if limit < len(classes):
        raise ValueError(
            "limit_samples must be at least the number of represented classes"
        )

    quotas = np.ones(len(classes), dtype=int)
    remaining = limit - len(classes)
    while remaining:
        capacities = counts - quotas
        eligible = capacities > 0
        if not eligible.any():
            break
        weights = np.where(eligible, counts, 0)
        exact_quotas = weights * (remaining / weights.sum())
        additions = np.minimum(
            np.floor(exact_quotas).astype(int), capacities
        )
        assigned = int(additions.sum())
        quotas += additions
        remaining -= assigned
        if remaining:
            remainders = exact_quotas - np.floor(exact_quotas)
            order = np.argsort(-remainders, kind="stable")
            for class_index in order:
                if remaining == 0:
                    break
                if quotas[class_index] < counts[class_index]:
                    quotas[class_index] += 1
                    remaining -= 1

    rng = np.random.default_rng(seed)
    selected = []
    for label, quota in zip(classes, quotas):
        class_indices = indices[split_targets == label].copy()
        rng.shuffle(class_indices)
        selected.extend(class_indices[:quota].tolist())
    rng.shuffle(selected)
    return selected


def split_class_counts(targets, splits, class_names=None):
    """Return class counts for each named partition."""
    targets = np.asarray(targets)
    if class_names is None:
        class_names = [str(label) for label in np.unique(targets)]
    return {
        split_name: {
            str(class_names[label]): int(np.count_nonzero(targets[indices] == label))
            for label in range(len(class_names))
        }
        for split_name, indices in splits.items()
    }


def save_test_metrics(output_dir: str, metrics: dict, metadata: dict) -> str:
    """Write the final held-out test metrics as JSON and a one-row CSV."""
    os.makedirs(output_dir, exist_ok=True)
    report = {"metadata": metadata, "metrics": metrics}
    json_path = os.path.join(output_dir, "test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)

    csv_path = os.path.join(output_dir, "test_metrics.csv")
    row = {
        **metadata,
        **{
            key: json.dumps(value) if isinstance(value, (list, dict)) else value
            for key, value in metrics.items()
        },
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return json_path


def build_ansatz(config):
    """Construct the ansatz instance from the config 'ansatz' key."""
    ansatz_type = config.get("ansatz", "dense")
    if ansatz_type == "dense":
        return DenseQCNNAnsatz4NoPool()
    return SingleAxisQCNNAnsatz4NoPool(rotation_gate=ansatz_type)


def setup_logger(output_dir: str, logger_name: str = "training") -> logging.Logger:
    """Create a logger that writes to both a file and stdout.

    Args:
        output_dir: Directory where training.log will be written
        logger_name: Name prefix for the logger (e.g., 'train_mnist')

    Returns:
        Configured logger instance
    """
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f"{logger_name}.{id(output_dir)}")
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


def save_confusion_matrix(output_dir: str, confusion_matrix, class_labels) -> str:
    """Save a labeled confusion matrix CSV and return its path.

    Args:
        output_dir: Directory where confusion_matrix_best.csv will be written
        confusion_matrix: NumPy array of shape (n_classes, n_classes)
        class_labels: List of class label strings

    Returns:
        Path to the saved confusion matrix CSV file
    """
    path = os.path.join(output_dir, "confusion_matrix_best.csv")
    class_labels = [str(label) for label in class_labels]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", *class_labels])
        for idx, row in enumerate(confusion_matrix.astype(int)):
            writer.writerow([class_labels[idx], *row.tolist()])

    return path
