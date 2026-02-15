"""
Headless trainer for queue-based HPC systems (SLURM, PBS, etc.).

Extends the base Trainer with:
- File-based logging (training.log) instead of tqdm/print
- CSV metrics export (metrics.csv) per epoch
- Model checkpointing per epoch + best model tracking
- Config snapshot (config.json) for reproducibility

All output goes to a single output directory, inspectable from a login node.
"""

import csv
import json
import logging
import os
import sys
import time
import torch

from ..notebooks.trainer import Trainer


class HeadlessTrainer(Trainer):
    """
    Trainer subclass designed for headless/queue environments.

    Replaces tqdm progress bars and print() with file logging,
    and adds automatic CSV metrics, checkpointing, and best-model tracking.
    """

    def __init__(
        self,
        criterion,
        device,
        output_dir,
        max_grad_norm=None,
        log_interval=10,
        save_every=1,
    ):
        """
        Args:
            criterion: Loss function
            device: torch.device for computation
            output_dir: Directory for logs, metrics CSV, and checkpoints
            max_grad_norm: Max gradient norm for clipping (None to disable)
            log_interval: Log every N batches
            save_every: Save checkpoint every N epochs (0 to only save final)
        """
        super().__init__(criterion, device, max_grad_norm, log_interval)
        self.output_dir = output_dir
        self.save_every = save_every
        os.makedirs(output_dir, exist_ok=True)

        # File logger
        self.logger = self._setup_logger()

        # CSV metrics
        self.metrics_path = os.path.join(output_dir, "metrics.csv")
        self._init_csv()

    def _setup_logger(self):
        log_path = os.path.join(self.output_dir, "training.log")
        logger = logging.getLogger(f"headless_trainer.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

        return logger

    _CSV_FIELDS = [
        "epoch", "train_loss", "train_acc",
        "test_loss", "test_acc", "epoch_time_s", "lr",
    ]

    def _init_csv(self):
        with open(self.metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._CSV_FIELDS).writeheader()

    def _log_csv(self, row: dict):
        with open(self.metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._CSV_FIELDS).writerow(row)

    def _save_checkpoint(self, epoch, model, optimizer, metrics, filename=None):
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

    def save_config(self, config: dict):
        """Dump the full configuration to config.json."""
        path = os.path.join(self.output_dir, "config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Override train() to add logging, CSV, and checkpointing

    def train(
        self, model, train_loader, optimizer, epochs, test_loader=None, scheduler=None
    ) -> dict[str, list]:
        """
        Train with file-based logging, CSV metrics, and checkpointing.
        Same interface as Trainer.train().
        """
        model.to(self.device)
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        best_test_acc = 0.0

        self.logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train one epoch
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm
                    )

                optimizer.step()

                running_loss += loss.item() * images.size(0)
                with torch.no_grad():
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).long()
                    correct += (preds == labels.long()).sum().item()
                    total += labels.size(0)

                if batch_idx % self.log_interval == 0:
                    acc = correct / total if total > 0 else 0
                    self.logger.info(
                        f"  Epoch {epoch}/{epochs} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f} | Acc: {acc:.4f}"
                    )

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Evaluate
            test_loss = test_acc = None
            cm: dict = {}
            if test_loader is not None:
                (test_loss, test_acc), confusion_matrix = self.evaluate(
                    model, test_loader
                )
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)
                tn, fp, fn, tp = confusion_matrix.ravel()
                cm = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

            # Scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(
                        test_loss if test_loss is not None else epoch_train_loss
                    )
                else:
                    scheduler.step()

            # Logging & metrics
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]

            csv_row = {
                "epoch": epoch,
                "train_loss": f"{epoch_train_loss:.6f}",
                "train_acc": f"{epoch_train_acc:.6f}",
                "test_loss": f"{test_loss:.6f}" if test_loss is not None else "",
                "test_acc": f"{test_acc:.6f}" if test_acc is not None else "",
                "epoch_time_s": f"{epoch_time:.1f}",
                "lr": f"{current_lr:.6f}",
            }
            self._log_csv(csv_row)

            log_msg = (
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}"
            )
            if test_loss is not None:
                log_msg += (
                    f" | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
                    f" | CM: TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}"
                )
            log_msg += f" | Time: {epoch_time:.1f}s"
            self.logger.info(log_msg)

            # Checkpointing
            if self.save_every > 0 and epoch % self.save_every == 0:
                ckpt = self._save_checkpoint(epoch, model, optimizer, csv_row)
                self.logger.info(f"Checkpoint saved: {ckpt}")

            if test_acc is not None and test_acc > best_test_acc:
                best_test_acc = test_acc
                self._save_checkpoint(epoch, model, optimizer, csv_row, "best_model.pt")
                self.logger.info(
                    f"New best model (acc={test_acc:.4f}) saved to best_model.pt"
                )

        self._save_checkpoint(epochs, model, optimizer, csv_row, "final_model.pt")
        self.logger.info(f"Training complete. Best test accuracy: {best_test_acc:.4f}")
        self.logger.info(f"All outputs saved to: {self.output_dir}")

        if test_loader is not None:
            return {
                "train_loss": train_losses,
                "train_acc": train_accuracies,
                "test_loss": test_losses,
                "test_acc": test_accuracies,
            }
        return {
            "train_loss": train_losses,
            "train_acc": train_accuracies,
        }
