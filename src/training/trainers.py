import torch

from .base import BaseTrainer


class BinaryTrainer(BaseTrainer):
    def _evaluate_batch(self, outputs, labels) -> tuple[torch.Tensor, int, int]:
        # Get predicted probabilities for the positive class
        probs = torch.sigmoid(outputs).squeeze()
        # Convert probabilities to binary predictions (threshold at 0.5)
        preds = (probs >= 0.5).long()
        # Calculate number of correct predictions and total samples
        batch_correct = (preds == labels).sum().item()
        batch_total = labels.size(0)
        return preds, batch_correct, batch_total

    def evaluate(self, model, test_loader):
        """Evaluate with accuracy, precision, recall, and F1."""
        metrics, confusion_matrix = super().evaluate(model, test_loader)

        tn = confusion_matrix[0, 0]  # noqa: F841
        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]
        tp = confusion_matrix[1, 1]

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

        return metrics, confusion_matrix


class MultiClassTrainer(BaseTrainer):
    def _evaluate_batch(self, outputs, labels) -> tuple[torch.Tensor, int, int]:
        # Get predicted class indices
        preds = torch.argmax(outputs, dim=1)
        # Calculate number of correct predictions and total samples
        batch_correct = (preds == labels).sum().item()
        batch_total = labels.size(0)
        return preds, batch_correct, batch_total


class MultiLabelTrainer(BaseTrainer):
    """Trainer for multi-label classification using BCEWithLogitsLoss.

    Accuracy is computed as *subset accuracy* (exact match ratio).
    Every label for a sample must be correct for it to count.
    """

    def _evaluate_batch(self, outputs, labels) -> tuple[torch.Tensor, int, int]:
        # Multi-label: threshold at 0 (logits)
        preds = (outputs > 0).float()
        # Subset accuracy: for each sample, ALL labels must match
        batch_correct = (preds == labels).all(dim=1).sum().item()
        batch_total = labels.size(0)
        return preds, batch_correct, batch_total

    def evaluate(self, model, test_loader):
        """Evaluate with subset accuracy, per-class F1, Micro F1, and Macro F1."""
        import numpy as np

        model.to(self.device)
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self._wrap_loader(
                test_loader, desc="Evaluating",
            ):
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                outputs = model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)

                preds, batch_correct, batch_total = self._evaluate_batch(
                    outputs, labels,
                )
                correct += batch_correct
                total += batch_total

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / total
        subset_acc = correct / total

        # Stack into (N, C) arrays
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Per-class TP, FP, FN
        tp = ((all_preds == 1) & (all_labels == 1)).sum(axis=0)
        fp = ((all_preds == 1) & (all_labels == 0)).sum(axis=0)
        fn = ((all_preds == 0) & (all_labels == 1)).sum(axis=0)

        # Per-class F1 (zero-safe)
        denom = 2 * tp + fp + fn
        per_class_f1 = np.where(denom > 0, 2 * tp / denom, 0.0)

        # Macro F1: mean of per-class F1
        macro_f1 = float(per_class_f1.mean())

        # Micro F1: pool TP/FP/FN across all classes
        tp_sum, fp_sum, fn_sum = tp.sum(), fp.sum(), fn.sum()
        micro_denom = 2 * tp_sum + fp_sum + fn_sum
        micro_f1 = float(2 * tp_sum / micro_denom) if micro_denom > 0 else 0.0

        metrics = {
            "loss": avg_loss,
            "acc": subset_acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1.tolist(),
        }

        return metrics, np.zeros((1, 1), dtype=int)
