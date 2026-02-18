from abc import ABC, abstractmethod
from collections.abc import Iterable
import csv
import json
import logging
import os
import time

import numpy as np
import torch
from tqdm import tqdm


class BaseTrainer(ABC):
    """
    Abstract base class for training quantum machine learning models.
    Subclasses should implement specific training algorithms and strategies.

    Pass a ``logging.Logger`` to get file/stream logging (headless, HPC).
    Omit it to get interactive ``tqdm`` progress bars (notebooks, terminals).
    """

    _CSV_FIELDS = [
        "epoch", "train_loss", "train_acc", "test_loss", "test_acc",
        "epoch_time_s", "lr",
    ]

    def __init__(
        self,
        criterion,
        device,
        max_grad_norm=None,
        log_interval=10,
        logger: logging.Logger | None = None,
        output_dir: str | None = None,
        save_every: int = 1,
    ) -> None:
        """
        Initialize trainer with configuration.

        Args:
            criterion: Loss function
            device: torch.device for computation
            max_grad_norm: Maximum gradient norm for clipping (None to disable).
                          Recommended: 1.0 for quantum models to handle noisy gradients
            log_interval: Interval for logging progress during training
            logger: Optional logger. When provided, progress is reported
                    via ``logger.info()`` instead of ``tqdm`` progress bars.
            output_dir: Optional directory for CSV metrics, checkpoints, and
                       config snapshots. When ``None`` these features are disabled.
            save_every: Save a checkpoint every N epochs (0 to only save final).
                       Only takes effect when *output_dir* is set.
        """
        self.criterion = criterion
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self._logger = logger
        self.output_dir = output_dir
        self.save_every = save_every

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            self._metrics_path = os.path.join(output_dir, "metrics.csv")
            self._init_csv()

    def _wrap_loader(self, data_loader, desc: str) -> Iterable:
        if self._logger is not None:
            self._loader_desc = desc
            self._loader_len = len(data_loader)
            return data_loader
        return tqdm(data_loader, desc=desc)

    def _report_batch(self, loop, batch_idx: int, **metrics) -> None:
        if self._logger is not None:
            parts = " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            )
            self._logger.info(
                f"  {self._loader_desc} | "
                f"Batch {batch_idx}/{self._loader_len} | {parts}"
            )
        elif hasattr(loop, "set_postfix"):
            loop.set_postfix(**metrics)

    def _report_epoch(self, message: str) -> None:
        if self._logger is not None:
            self._logger.info(message)
        else:
            print(message)

    def _init_csv(self) -> None:
        with open(self._metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._CSV_FIELDS).writeheader()

    def _log_csv(self, row: dict) -> None:
        with open(self._metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._CSV_FIELDS).writerow(row)

    def _save_checkpoint(
        self, epoch, model, optimizer, metrics, filename=None,
    ) -> str:
        assert self.output_dir is not None
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        path = os.path.join(self.output_dir, filename)
        torch.save(state, path)
        return path

    def save_config(self, config: dict) -> None:
        """Dump the full run configuration to ``config.json``.

        Only available when *output_dir* is set.
        """
        if self.output_dir is None:
            return
        path = os.path.join(self.output_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        self._report_epoch(f"Configuration saved to {path}")

    @abstractmethod
    def _evaluate_batch(self, outputs, labels) -> tuple[torch.Tensor, int, int]:
        pass

    def train(
        self, model, train_loader, optimizer, epochs, test_loader=None, scheduler=None
    ) -> dict[str, list]:
        """
        Train the given model using the provided data loader and optimizer.

        Args:
            model: The quantum machine learning model to be trained.
            train_loader: An iterable that provides batches of training data.
            optimizer: The optimization algorithm to update model parameters.
            epochs: The number of epochs to train the model.
            test_loader: An optional iterable that provides batches of test data.
            scheduler: An optional learning rate scheduler.
        """
        model.to(self.device)
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        best_test_acc = 0.0

        self._report_epoch(f"Starting training for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            loop = self._wrap_loader(
                train_loader, desc=f"Epoch {epoch}/{epochs}"
            )

            for batch_idx, (images, labels) in enumerate(loop):
                images = images.to(self.device)
                labels = labels.to(self.device)
                if isinstance(self.criterion, (
                    torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss,
                )):
                    labels = labels.float()
                else:
                    labels = labels.long()

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.max_grad_norm is not None:
                    # Gradient clipping for stability (helps with noisy quantum
                    # gradients)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm
                    )

                optimizer.step()

                running_loss += loss.item() * images.size(0)

                with torch.no_grad():
                    _, batch_correct, batch_total = self._evaluate_batch(
                        outputs, labels
                    )
                    correct += batch_correct
                    total += batch_total

                if batch_idx % self.log_interval == 0:
                    current_acc = correct / total if total > 0 else 0
                    self._report_batch(
                        loop, batch_idx,
                        loss=loss.item(), acc=current_acc,
                    )

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total

            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Run evaluation if test_loader provided
            test_loss = test_acc = None
            if test_loader is not None:
                (test_loss, test_acc), _ = self.evaluate(model, test_loader)
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)
                self._report_epoch(
                    f"Epoch {epoch}: Train Loss={epoch_train_loss:.4f}, "
                    f"Train Acc={epoch_train_acc:.4f} | "
                    f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}"
                )
            else:
                self._report_epoch(
                    f"Epoch {epoch}: Loss={epoch_train_loss:.4f}, "
                    f"Acc={epoch_train_acc:.4f}"
                )

            # Step the scheduler if provided
            if scheduler is not None:
                # If scheduler is ReduceLROnPlateau, pass validation loss
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = (
                        test_loss
                        if test_loss is not None
                        else epoch_train_loss
                    )
                    scheduler.step(metric)
                else:
                    scheduler.step()

            # CSV + checkpointing (only when output_dir is set)
            if self.output_dir is not None:
                epoch_time = time.time() - epoch_start
                current_lr = optimizer.param_groups[0]["lr"]
                csv_row = {
                    "epoch": epoch,
                    "train_loss": f"{epoch_train_loss:.6f}",
                    "train_acc": f"{epoch_train_acc:.6f}",
                    "test_loss": (
                        f"{test_loss:.6f}" if test_loss is not None else ""
                    ),
                    "test_acc": (
                        f"{test_acc:.6f}" if test_acc is not None else ""
                    ),
                    "epoch_time_s": f"{epoch_time:.1f}",
                    "lr": f"{current_lr:.6f}",
                }
                self._log_csv(csv_row)

                if self.save_every > 0 and epoch % self.save_every == 0:
                    ckpt = self._save_checkpoint(
                        epoch, model, optimizer, csv_row,
                    )
                    self._report_epoch(f"Checkpoint saved: {ckpt}")

                if test_acc is not None and test_acc > best_test_acc:
                    best_test_acc = test_acc
                    self._save_checkpoint(
                        epoch, model, optimizer, csv_row, "best_model.pt",
                    )
                    self._report_epoch(
                        f"New best model (acc={test_acc:.4f}) "
                        f"saved to best_model.pt"
                    )

        # Final checkpoint (skip if the last epoch was already saved)
        if self.output_dir is not None:
            already_saved = (
                self.save_every > 0 and epochs % self.save_every == 0
            )
            if not already_saved:
                self._save_checkpoint(
                    epochs, model, optimizer, csv_row, "final_model.pt",
                )
            self._report_epoch(
                f"Training complete. Best test acc: {best_test_acc:.4f}"
            )
            self._report_epoch(f"All outputs saved to: {self.output_dir}")

        # Return results
        result = {
            "train_loss": train_losses,
            "train_acc": train_accuracies,
        }

        if test_loader is not None:
            result["test_loss"] = test_losses
            result["test_acc"] = test_accuracies

        return result

    def evaluate(self, model, test_loader) -> tuple[tuple[float, float], np.ndarray]:
        """
        Evaluate the model on the test data.

        Args:
            model: The trained model to be evaluated.
            test_loader: An iterable that provides batches of test data.

        Returns:
            A tuple containing:
                - A tuple of (test_loss, test_accuracy)
                - A numpy array of predictions for the test set
        """
        model.to(self.device)
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # For confusion matrix: [TN, FP, FN, TP]
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self._wrap_loader(
                test_loader, desc="Evaluating",
            ):
                images = images.to(self.device)
                labels = labels.to(self.device)
                if isinstance(self.criterion, (
                    torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss,
                )):
                    labels = labels.float()
                else:
                    labels = labels.long()

                outputs = model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)

                # Get predictions for binary classification
                preds, batch_correct, batch_total = self._evaluate_batch(
                    outputs, labels
                )
                correct += batch_correct
                total += batch_total

                # Store for confusion matrix
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.long().cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / total
        accuracy = correct / total

        # Build confusion matrix
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        tn = np.sum((all_labels == 0) & (all_preds == 0))
        fp = np.sum((all_labels == 0) & (all_preds == 1))
        fn = np.sum((all_labels == 1) & (all_preds == 0))
        tp = np.sum((all_labels == 1) & (all_preds == 1))

        confusion_matrix = np.array([[tn, fp], [fn, tp]])

        return (avg_loss, accuracy), confusion_matrix
