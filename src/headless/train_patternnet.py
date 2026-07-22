"""
Headless training script for PatternNet remote-sensing scene classification.
Designed for queue-based HPC systems (SLURM, PBS, etc.).

PatternNet must be extracted into class-named folders, for example:
    src/data/patternnet/airplane/*.jpg
    src/data/patternnet/baseball_field/*.jpg

The dataset has no official train/validation/test split, so this script creates
a reproducible stratified 70%/15%/15% split for every class.

Outputs:
    <output_dir>/
        metrics.csv                  - Per-epoch train/validation metrics
        training.log                 - Detailed log with timestamps
        checkpoint_epoch_N.pt        - Model checkpoint per epoch
        best_model.pt                - Best model by validation accuracy
        final_model.pt               - Final model state dict
        config.json                  - Full training configuration
        confusion_matrix_best.csv    - Test confusion matrix of best model

Usage:
    python -m src.headless.train_patternnet
    python -m src.headless.train_patternnet --data-root data/PatternNet
"""

import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ..qml.models.multiclass import MultiClassCNN, MultiClassQCNN
from ..training.trainers import MultiClassTrainer
from ..training.shared import (
    set_seed,
    seed_worker,
    build_ansatz,
    setup_logger,
    save_confusion_matrix
)


PATTERNNET_DOWNLOAD_URL = (
    "https://nuisteducn1-my.sharepoint.com/:u:/g/personal/zhouwx_nuist_edu_cn/"
    "EYSPYqBztbBBqS27B7uM_mEB3R9maNJze8M1Qg9Q6cnPBQ?e=MSf977&download=1"
)


CONFIG = {
    # Data
    "data_root": "src/data/patternnet",
    "image_size": 64,
    "train_fraction": 0.70,
    "val_fraction": 0.15,
    "limit_samples": None,
    # Model
    "num_classes": 38,
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
    "output_dir": "runs/patternnet",
    "log_interval": 20,
    "save_every": 1,
}


def _dataset_present(data_root: str, num_classes: int) -> bool:
    """Return True when data_root contains at least num_classes sub-directories."""
    if not os.path.isdir(data_root):
        return False
    class_dirs = [
        entry for entry in os.scandir(data_root)
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    return len(class_dirs) >= num_classes


def _download_with_progress(url: str, dest: str) -> None:
    """Download url to dest, printing a simple ASCII progress bar.

    Uses ``requests`` with a browser User-Agent to satisfy SharePoint's
    redirect/auth requirements. Falls back to ``urllib`` if ``requests`` is
    not installed, though that path is less likely to succeed against
    SharePoint.
    """
    _BROWSER_UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    try:
        import requests as _requests

        with _requests.get(
            url,
            headers={"User-Agent": _BROWSER_UA},
            stream=True,
            allow_redirects=True,
            timeout=60,
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as file:
                for chunk in response.iter_content(chunk_size=65536):
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        percent = min(downloaded * 100 // total, 100)
                        bar = "#" * (percent // 2)
                        print(
                            f"\r  [{bar:<50}] {percent:3d}%",
                            end="",
                            flush=True,
                        )
        print()
        return

    except ImportError:
        pass  # fall through to urllib

    import urllib.request

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", _BROWSER_UA)]
    urllib.request.install_opener(opener)

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = min(block_num * block_size, total_size)
            percent = downloaded * 100 // total_size
            bar = "#" * (percent // 2)
            print(f"\r  [{bar:<50}] {percent:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _flatten_single_subdir(directory: str) -> None:
    """Move contents up while there is exactly one sub-directory.

    Materialises the child list before moving to avoid iterator invalidation
    on Windows when items are removed from the directory mid-scan.
    """
    import shutil

    while True:
        entries = [
            entry for entry in os.scandir(directory)
            if entry.is_dir() and not entry.name.startswith(".")
        ]
        if len(entries) != 1:
            break
        sub = entries[0].path
        children = list(os.scandir(sub))  # materialise before any moves
        for item in children:
            shutil.move(item.path, directory)
        os.rmdir(sub)


def ensure_dataset(data_root: str, num_classes: int = CONFIG["num_classes"]) -> None:
    """Check that PatternNet class folders are present; download and extract if not.

    Requires internet access on first run. The zip (~1.5 GB) is fetched from the
    official SharePoint mirror and extracted in-place. If auto-download fails
    (e.g. the sharing token has expired), a clear error with the manual download
    URL is raised.
    """
    import zipfile

    if _dataset_present(data_root, num_classes):
        return

    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, "_patternnet_download.zip")

    print(
        f"PatternNet not found at '{data_root}'. Downloading (~1.5 GB) ..."
    )
    try:
        _download_with_progress(PATTERNNET_DOWNLOAD_URL, zip_path)
    except Exception as exc:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise RuntimeError(
            f"Auto-download failed: {exc}\n"
            "The sharing token in PATTERNNET_DOWNLOAD_URL may have expired.\n"
            "Download PatternNet manually from:\n"
            "  https://sites.google.com/view/zhouwx/dataset\n"
            f"Then extract it so that class folders appear directly inside "
            f"'{data_root}'."
        ) from exc

    print(f"Extracting to '{data_root}' ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)
    os.remove(zip_path)

    # PatternNet zip typically extracts into a single 'PatternNet/' sub-folder.
    _flatten_single_subdir(data_root)

    if not _dataset_present(data_root, num_classes):
        raise RuntimeError(
            f"Extraction complete but fewer than {num_classes} class folders were "
            f"found in '{data_root}'. Check the extracted directory structure."
        )
    print("PatternNet dataset ready.")


def parse_cli_overrides():
    """Allow overriding key run settings from the CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Quantum CNN on the PatternNet dataset"
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="PatternNet directory containing class folders")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--limit-samples", type=int, default=None,
                        help="Limit total dataset size for quick validation")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Override square resize/crop size")
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
        "data_root",
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

    # Auto-build output dir from config when not explicitly provided
    if args.output_dir is None:
        if config.get("use_classical", False):
            config["output_dir"] = "runs/patternnet_classical"
        else:
            abbrev = {"dense": "d", "rx": "rx", "ry": "ry", "rz": "rz"}
            enc = abbrev.get(config["encoding"], config["encoding"])
            ans = abbrev.get(config["ansatz"], config["ansatz"])
            meas = config["measurement"]
            config["output_dir"] = f"runs/patternnet_{enc}_{ans}_{meas}"
    else:
        config["output_dir"] = args.output_dir

    return config


def split_indices_by_class(targets, config):
    """Create deterministic, stratified train/validation/test indices."""
    targets = np.asarray(targets)
    rng = np.random.default_rng(config["seed"])
    train_indices, val_indices, test_indices = [], [], []

    for class_index in np.unique(targets):
        class_indices = np.flatnonzero(targets == class_index)
        rng.shuffle(class_indices)
        count = len(class_indices)
        train_count = int(count * config["train_fraction"])
        val_count = int(count * config["val_fraction"])
        test_count = count - train_count - val_count
        if min(train_count, val_count, test_count) < 1:
            raise ValueError(
                "Each PatternNet class needs enough samples for train, "
                "validation, and test splits."
            )
        train_indices.extend(class_indices[:train_count].tolist())
        val_indices.extend(class_indices[train_count:train_count + val_count].tolist())
        test_indices.extend(class_indices[train_count + val_count:].tolist())

    return train_indices, val_indices, test_indices


def limit_split_indices(indices, limit, seed):
    """Deterministically limit a split while preserving reproducible ordering."""
    if limit is None or len(indices) <= limit:
        return indices
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(indices), generator=generator)[:limit].tolist()
    return [indices[index] for index in selected]


def load_data(config, use_cuda: bool):
    """Load PatternNet class folders and create stratified dataset splits."""
    ensure_dataset(config["data_root"], config["num_classes"])

    patternnet_mean = (0.485, 0.456, 0.406)
    patternnet_std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            config["image_size"], scale=(0.7, 1.0), ratio=(0.8, 1.25)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03
        ),
        transforms.ToTensor(),
        transforms.Normalize(patternnet_mean, patternnet_std),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(patternnet_mean, patternnet_std),
    ])

    train_dataset_full = ImageFolder(config["data_root"], transform=train_transform)
    evaluation_dataset_full = ImageFolder(
        config["data_root"], transform=evaluation_transform
    )
    if train_dataset_full.classes != evaluation_dataset_full.classes:
        raise RuntimeError("PatternNet dataset views have inconsistent class mappings")
    if len(train_dataset_full.classes) != config["num_classes"]:
        raise ValueError(
            f"Expected {config['num_classes']} PatternNet classes, found "
            f"{len(train_dataset_full.classes)} in {config['data_root']}"
        )

    train_indices, val_indices, test_indices = split_indices_by_class(
        train_dataset_full.targets, config
    )
    limit = config["limit_samples"]
    if limit is not None:
        train_indices = limit_split_indices(train_indices, limit, config["seed"])
        evaluation_limit = max(limit // 5, 1)
        val_indices = limit_split_indices(
            val_indices, evaluation_limit, config["seed"] + 1
        )
        test_indices = limit_split_indices(
            test_indices, evaluation_limit, config["seed"] + 2
        )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(evaluation_dataset_full, val_indices)
    test_dataset = Subset(evaluation_dataset_full, test_indices)

    pin_memory = use_cuda
    persistent_workers = config["num_workers"] > 0
    worker_init_fn = partial(seed_worker, base_seed=config["seed"])
    train_generator = torch.Generator().manual_seed(config["seed"])
    eval_generator = torch.Generator().manual_seed(config["seed"] + 1)

    loader_options = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": worker_init_fn,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=train_generator, **loader_options
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, generator=eval_generator, **loader_options
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, generator=eval_generator, **loader_options
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        train_dataset_full.classes,
    )


def build_model(config, device):
    """Construct the selected multiclass model."""
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

    (
        train_loader,
        val_loader,
        test_loader,
        n_train,
        n_val,
        n_test,
        classes,
    ) = load_data(config, use_cuda=(device.type == "cuda"))
    model = build_model(config, device)
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

    logger = setup_logger(config["output_dir"], logger_name="train_patternnet")
    trainer = MultiClassTrainer(
        criterion=criterion,
        device=device,
        max_grad_norm=config["max_grad_norm"],
        log_interval=config["log_interval"],
        logger=logger,
        output_dir=config["output_dir"],
        save_every=config["save_every"],
    )
    config["device"] = str(device)
    config["classes"] = classes
    trainer.save_config(config)
    logger.info(
        f"Train samples: {n_train}, Val samples: {n_val}, Test samples: {n_test}"
    )
    logger.info(f"Classes ({len(classes)}): {classes[:5]}...")
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    logger.info(
        f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
    )

    trainer.train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=config["epochs"],
        test_loader=val_loader,
        scheduler=scheduler,
    )

    best_model_path = os.path.join(config["output_dir"], "best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best validation model from {best_model_path}")
    else:
        logger.info("best_model.pt not found; using final model for evaluation")

    test_metrics, confusion_matrix = trainer.evaluate(model, test_loader)
    confusion_matrix_path = save_confusion_matrix(
        config["output_dir"], confusion_matrix, classes
    )
    logger.info(
        "Final Test | "
        f"Loss={test_metrics['loss']:.4f}, Acc={test_metrics['acc']:.4f}"
    )
    logger.info(f"Confusion matrix saved to {confusion_matrix_path}")


if __name__ == "__main__":
    main()
