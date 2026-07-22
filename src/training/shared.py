"""
Shared utility functions for headless training scripts.
"""

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
    import csv

    path = os.path.join(output_dir, "confusion_matrix_best.csv")
    class_labels = [str(label) for label in class_labels]
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", *class_labels])
        for idx, row in enumerate(confusion_matrix.astype(int)):
            writer.writerow([class_labels[idx], *row.tolist()])

    return path
