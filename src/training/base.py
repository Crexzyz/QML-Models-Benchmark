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
        "precision", "recall", "f1", "micro_f1", "macro_f1",
        "train_correct", "train_total", "test_correct", "test_total",
        "per_class_precision", "per_class_recall", "per_class_f1",
        "epoch_time_s", "lr_before_scheduler", "lr", "scheduler_reduced_lr",
        "gpu_peak_memory_allocated_bytes",
        "train_quantum_forward_calls", "train_quantum_circuit_executions",
        "train_quantum_patches",
        "train_quantum_layer_forward_time_s", "train_quantum_circuit_time_s",
        "test_quantum_forward_calls", "test_quantum_circuit_executions",
        "test_quantum_patches",
        "test_quantum_layer_forward_time_s", "test_quantum_circuit_time_s",
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
            csv.DictWriter(f, fieldnames=self._CSV_FIELDS, restval="").writeheader()

    def _log_csv(self, row: dict) -> None:
        with open(self._metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._CSV_FIELDS, restval="").writerow(row)

    @staticmethod
    def _reset_quantum_cost_metrics(model) -> None:
        for module in model.modules():
            reset_metrics = getattr(module, "reset_cost_metrics", None)
            if callable(reset_metrics):
                reset_metrics()

    @staticmethod
    def _collect_quantum_cost_metrics(model) -> dict:
        metrics = {
            "forward_calls": 0,
            "circuit_executions": 0,
            "patches": 0,
            "layer_forward_time_s": 0.0,
            "circuit_time_s": 0.0,
        }

        for module in model.modules():
            get_metrics = getattr(module, "get_cost_metrics", None)
            if not callable(get_metrics):
                continue
            module_metrics = get_metrics()
            for field in (
                "forward_calls",
                "circuit_executions",
                "patches",
                "layer_forward_time_s",
                "circuit_time_s",
            ):
                metrics[field] += module_metrics[field]
        return metrics

    def _save_summary(self, epoch_records: list[dict]) -> None:
        """Save final and derived metrics for a single training run."""
        if self.output_dir is None or not epoch_records:
            return

        final_metrics = epoch_records[-1]
        test_records = [
            record for record in epoch_records if record["test_acc"] is not None
        ]
        best_metrics = (
            max(test_records, key=lambda record: record["test_acc"])
            if test_records
            else None
        )

        summary = {
            "epochs_completed": len(epoch_records),
            "total_training_time_s": sum(
                record["epoch_time_s"] for record in epoch_records
            ),
            "mean_epoch_time_s": sum(
                record["epoch_time_s"] for record in epoch_records
            ) / len(epoch_records),
            "scheduler_lr_reductions": sum(
                record["scheduler_reduced_lr"] for record in epoch_records
            ),
            "final": final_metrics,
        }
        quantum_cost = {}
        for phase in ("train", "test"):
            total_patches = sum(
                record[f"{phase}_quantum_patches"] for record in epoch_records
            )
            total_layer_time = sum(
                record[f"{phase}_quantum_layer_forward_time_s"]
                for record in epoch_records
            )
            total_circuit_time = sum(
                record[f"{phase}_quantum_circuit_time_s"]
                for record in epoch_records
            )
            quantum_cost[phase] = {
                "forward_calls": sum(
                    record[f"{phase}_quantum_forward_calls"]
                    for record in epoch_records
                ),
                "circuit_executions": sum(
                    record[f"{phase}_quantum_circuit_executions"]
                    for record in epoch_records
                ),
                "patches": total_patches,
                "layer_forward_time_s": total_layer_time,
                "circuit_time_s": total_circuit_time,
                "layer_forward_time_per_patch_s": (
                    total_layer_time / total_patches if total_patches else None
                ),
                "circuit_time_per_patch_s": (
                    total_circuit_time / total_patches if total_patches else None
                ),
                "layer_forward_time_per_call_s": (
                    total_layer_time
                    / sum(
                        record[f"{phase}_quantum_forward_calls"]
                        for record in epoch_records
                    )
                    if total_patches
                    else None
                ),
                "layer_forward_time_per_image_s": (
                    total_layer_time
                    / sum(record[f"{phase}_total"] for record in epoch_records)
                    if total_patches
                    else None
                ),
            }
        summary["quantum_cost"] = quantum_cost
        if best_metrics is not None:
            min_loss_metrics = min(
                test_records, key=lambda record: record["test_loss"]
            )
            summary["best_validation"] = best_metrics
            summary["best_epoch"] = best_metrics["epoch"]
            summary["time_to_best_epoch_s"] = sum(
                record["epoch_time_s"]
                for record in epoch_records[:best_metrics["epoch"]]
            )
            summary["final_generalization_gap"] = (
                final_metrics["train_acc"] - final_metrics["test_acc"]
            )
            summary["best_generalization_gap"] = (
                best_metrics["train_acc"] - best_metrics["test_acc"]
            )
            summary["minimum_test_loss"] = min_loss_metrics["test_loss"]
            summary["minimum_test_loss_epoch"] = min_loss_metrics["epoch"]

        path = os.path.join(self.output_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        self._report_epoch(f"Training summary saved to {path}")

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

    def _compute_confusion_matrix(
        self,
        all_labels: np.ndarray,
        all_preds: np.ndarray,
    ) -> np.ndarray | None:
        """Optional confusion matrix hook for single-label tasks."""
        return None

    def _compute_additional_metrics(
        self,
        confusion_matrix: np.ndarray | None,
    ) -> dict:
        """Optional additional metrics derived from evaluation outputs."""
        return {}

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
        epoch_records = []

        self._report_epoch(f"Starting training for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self._reset_quantum_cost_metrics(model)
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            loop = self._wrap_loader(
                train_loader, desc=f"Epoch {epoch}/{epochs}"
            )

            for batch_idx, (images, labels) in enumerate(loop):
                non_blocking = self.device.type == "cuda"
                images = images.to(self.device, non_blocking=non_blocking)
                labels = labels.to(self.device, non_blocking=non_blocking)
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
            train_quantum_metrics = self._collect_quantum_cost_metrics(model)

            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Run evaluation if test_loader provided
            test_loss = test_acc = None
            test_metrics: dict = {}
            if test_loader is not None:
                self._reset_quantum_cost_metrics(model)
                test_metrics, _ = self.evaluate(model, test_loader)
                test_quantum_metrics = self._collect_quantum_cost_metrics(model)
                test_loss = test_metrics["loss"]
                test_acc = test_metrics["acc"]
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)
                extra = ""
                if "f1" in test_metrics:
                    extra = f", F1={test_metrics['f1']:.4f}"
                elif "micro_f1" in test_metrics:
                    extra = (
                        f", MicroF1={test_metrics['micro_f1']:.4f}"
                        f", MacroF1={test_metrics['macro_f1']:.4f}"
                    )
                self._report_epoch(
                    f"Epoch {epoch}: Train Loss={epoch_train_loss:.4f}, "
                    f"Train Acc={epoch_train_acc:.4f} | "
                    f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}{extra}"
                )
                if "per_class_f1" in test_metrics:
                    per_class = [f'{v:.4f}' for v in test_metrics['per_class_f1']]
                    self._report_epoch(f"  Per-class F1: {per_class}")
            else:
                test_quantum_metrics = {
                    "forward_calls": 0,
                    "circuit_executions": 0,
                    "patches": 0,
                    "layer_forward_time_s": 0.0,
                    "circuit_time_s": 0.0,
                }
                self._report_epoch(
                    f"Epoch {epoch}: Loss={epoch_train_loss:.4f}, "
                    f"Acc={epoch_train_acc:.4f}"
                )

            # Step the scheduler if provided
            lr_before_scheduler = optimizer.param_groups[0]["lr"]
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
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler_reduced_lr = current_lr < lr_before_scheduler

            # CSV + checkpointing (only when output_dir is set)
            if self.output_dir is not None:
                epoch_time = time.time() - epoch_start
                peak_memory = (
                    torch.cuda.max_memory_allocated(self.device)
                    if self.device.type == "cuda"
                    else ""
                )
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
                    "train_correct": correct,
                    "train_total": total,
                    "test_correct": test_metrics.get("correct", ""),
                    "test_total": test_metrics.get("total", ""),
                    "per_class_precision": json.dumps(
                        test_metrics.get("per_class_precision", [])
                    ),
                    "per_class_recall": json.dumps(
                        test_metrics.get("per_class_recall", [])
                    ),
                    "per_class_f1": json.dumps(
                        test_metrics.get("per_class_f1", [])
                    ),
                    "epoch_time_s": f"{epoch_time:.1f}",
                    "lr_before_scheduler": f"{lr_before_scheduler:.6f}",
                    "lr": f"{current_lr:.6f}",
                    "scheduler_reduced_lr": int(scheduler_reduced_lr),
                    "gpu_peak_memory_allocated_bytes": peak_memory,
                }
                for phase, quantum_metrics in (
                    ("train", train_quantum_metrics),
                    ("test", test_quantum_metrics),
                ):
                    csv_row.update({
                        f"{phase}_quantum_forward_calls": (
                            quantum_metrics["forward_calls"]
                        ),
                        f"{phase}_quantum_circuit_executions": (
                            quantum_metrics["circuit_executions"]
                        ),
                        f"{phase}_quantum_patches": quantum_metrics["patches"],
                        f"{phase}_quantum_layer_forward_time_s": (
                            f"{quantum_metrics['layer_forward_time_s']:.6f}"
                        ),
                        f"{phase}_quantum_circuit_time_s": (
                            f"{quantum_metrics['circuit_time_s']:.6f}"
                        ),
                    })
                for field in ("precision", "recall", "f1", "micro_f1", "macro_f1"):
                    if field in test_metrics:
                        csv_row[field] = f"{test_metrics[field]:.6f}"
                self._log_csv(csv_row)

                epoch_record = {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "train_acc": epoch_train_acc,
                    "train_correct": correct,
                    "train_total": total,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_correct": test_metrics.get("correct"),
                    "test_total": test_metrics.get("total"),
                    "macro_f1": test_metrics.get("macro_f1"),
                    "micro_f1": test_metrics.get("micro_f1"),
                    "epoch_time_s": epoch_time,
                    "lr_before_scheduler": lr_before_scheduler,
                    "lr": current_lr,
                    "scheduler_reduced_lr": scheduler_reduced_lr,
                    "gpu_peak_memory_allocated_bytes": peak_memory or None,
                }
                for phase, quantum_metrics in (
                    ("train", train_quantum_metrics),
                    ("test", test_quantum_metrics),
                ):
                    for field, value in quantum_metrics.items():
                        epoch_record[f"{phase}_quantum_{field}"] = value
                epoch_records.append(epoch_record)

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
            self._save_summary(epoch_records)
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

    def evaluate(self, model, test_loader) -> tuple[dict, np.ndarray]:
        """
        Evaluate the model on the test data.

        Args:
            model: The trained model to be evaluated.
            test_loader: An iterable that provides batches of test data.

        Returns:
            A tuple containing:
                - A dictionary with loss/accuracy and optional extra metrics
                - A confusion matrix when available (otherwise empty matrix)
        """
        model.to(self.device)
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # Collect predictions/labels for optional confusion matrix and
        # derived metrics.
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

                preds, batch_correct, batch_total = self._evaluate_batch(
                    outputs, labels
                )
                correct += batch_correct
                total += batch_total

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.long().cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / total
        accuracy = correct / total

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        confusion_matrix = self._compute_confusion_matrix(all_labels, all_preds)
        metrics = {
            "loss": avg_loss,
            "acc": accuracy,
            "correct": correct,
            "total": total,
        }
        metrics.update(self._compute_additional_metrics(confusion_matrix))

        if confusion_matrix is None:
            confusion_matrix = np.zeros((0, 0), dtype=int)

        return metrics, confusion_matrix
