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

    Accuracy is computed *per label* (i.e. each label slot counts as one
    prediction) which matches the notebook implementation.
    """

    def _evaluate_batch(self, outputs, labels) -> tuple[torch.Tensor, int, int]:
        # Multi-label: threshold at 0 (logits)
        preds = (outputs > 0).float()
        # Per-label accuracy: count every matching label entry
        batch_correct = (preds == labels).sum().item()
        batch_total = labels.numel()
        return preds, batch_correct, batch_total

    def evaluate(self, model, test_loader):
        """Evaluate with per-label accuracy (no confusion matrix)."""
        import numpy as np

        model.to(self.device)
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

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

        avg_loss = total_loss / total
        accuracy = correct / total

        # Multi-label: simple confusion matrix is not meaningful
        confusion_matrix = np.zeros((1, 1), dtype=int)
        return (avg_loss, accuracy), confusion_matrix
