import numpy as np
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

    def _compute_confusion_matrix(self, all_labels, all_preds):
        """Compute binary confusion matrix as [[TN, FP], [FN, TP]]."""
        tn = int(((all_labels == 0) & (all_preds == 0)).sum())
        fp = int(((all_labels == 0) & (all_preds == 1)).sum())
        fn = int(((all_labels == 1) & (all_preds == 0)).sum())
        tp = int(((all_labels == 1) & (all_preds == 1)).sum())
        return np.array([[tn, fp], [fn, tp]], dtype=int)

    def _compute_additional_metrics(self, confusion_matrix):
        """Compute precision, recall and F1 for binary classification."""
        if confusion_matrix is None or confusion_matrix.shape != (2, 2):
            return {}

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

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


class MultiClassTrainer(BaseTrainer):
    def _evaluate_batch(self, outputs, labels) -> tuple[torch.Tensor, int, int]:
        # Get predicted class indices
        preds = torch.argmax(outputs, dim=1)
        # Calculate number of correct predictions and total samples
        batch_correct = (preds == labels).sum().item()
        batch_total = labels.size(0)
        return preds, batch_correct, batch_total

    def _compute_confusion_matrix(self, all_labels, all_preds):
        """Compute multiclass confusion matrix with label index rows/cols."""
        if all_labels.size == 0:
            return np.zeros((0, 0), dtype=int)

        max_label = int(max(all_labels.max(initial=0), all_preds.max(initial=0)))
        num_classes = max_label + 1
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        np.add.at(confusion_matrix, (all_labels, all_preds), 1)
        return confusion_matrix

    def _compute_additional_metrics(self, confusion_matrix):
        """Compute macro/micro and per-class metrics for multiclass tasks."""
        if confusion_matrix is None or confusion_matrix.size == 0:
            return {}

        tp = np.diag(confusion_matrix).astype(float)
        predicted_per_class = confusion_matrix.sum(axis=0).astype(float)
        true_per_class = confusion_matrix.sum(axis=1).astype(float)

        per_class_precision = np.divide(
            tp,
            predicted_per_class,
            out=np.zeros_like(tp, dtype=float),
            where=predicted_per_class > 0,
        )
        per_class_recall = np.divide(
            tp,
            true_per_class,
            out=np.zeros_like(tp, dtype=float),
            where=true_per_class > 0,
        )
        per_class_f1 = np.divide(
            2.0 * per_class_precision * per_class_recall,
            per_class_precision + per_class_recall,
            out=np.zeros_like(tp, dtype=float),
            where=(per_class_precision + per_class_recall) > 0,
        )

        macro_precision = float(per_class_precision.mean())
        macro_recall = float(per_class_recall.mean())
        macro_f1 = float(per_class_f1.mean())

        tp_sum = float(tp.sum())
        total = float(confusion_matrix.sum())
        micro_f1 = float(tp_sum / total) if total > 0 else 0.0

        return {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "per_class_precision": per_class_precision.tolist(),
            "per_class_recall": per_class_recall.tolist(),
            "per_class_f1": per_class_f1.tolist(),
        }


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
