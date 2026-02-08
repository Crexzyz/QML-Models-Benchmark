"""
Trainer class for multiclass classification with quantum-classical hybrid models.
"""

import torch
import numpy as np
from tqdm import tqdm


class MultiClassTrainer:
    """
    Trainer class for multiclass classification with quantum-classical hybrid models.
    Handles training and evaluation with configurable settings.
    """

    def __init__(self, criterion, device, max_grad_norm=None, log_interval=10) -> None:
        """
        Initialize trainer with configuration.

        Args:
            criterion: Loss function (e.g., CrossEntropyLoss)
            device: torch.device for computation
            max_grad_norm: Maximum gradient norm for clipping (None to disable).
                          Recommended: 1.0 for quantum models to handle noisy gradients
            log_interval: Interval for logging progress during training
        """
        self.criterion = criterion
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval

    def train(
        self, model, train_loader, optimizer, epochs, test_loader=None, scheduler=None
    ) -> dict[str, list]:
        """
        Train the model.

        Args:
            model: Neural network model
            train_loader: DataLoader for training data
            optimizer: Optimizer
            epochs: Number of training epochs
            test_loader: Optional DataLoader for test data. If provided, evaluation
            runs after each epoch.
            scheduler: Optional learning rate scheduler.

        Returns:
            dict with keys:
                'train_loss': List of training losses per epoch
                'train_acc': List of training accuracies per epoch
                'test_loss': List of test losses per epoch (if test_loader provided)
                'test_acc': List of test accuracies per epoch (if test_loader provided)
        """
        model.to(self.device)
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

            for batch_idx, (images, labels) in enumerate(loop):
                images = images.to(self.device)

                # Check for float inputs => multi-label/BCE, else long for C-Entropy
                if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                    labels = labels.to(self.device).float()
                else:
                    labels = labels.to(self.device).long()

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
                    if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                        # Multi-label accuracy (Subset Accuracy aka Exact Match)
                        # Preds > 0 (since logits) means prob > 0.5
                        preds = (outputs > 0).float()
                        # correct if all classes match exactly
                        matches = (preds == labels).all(dim=1).float()
                        correct += matches.sum().item()
                    else:
                        # Single-label multiclass
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()

                    total += labels.size(0)

                # Update progress bar
                if batch_idx % self.log_interval == 0:
                    current_acc = correct / total if total > 0 else 0
                    loop.set_postfix(loss=loss.item(), acc=current_acc)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total

            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # Run evaluation if test_loader provided
            if test_loader is not None:
                (test_loss, test_acc), _ = self.evaluate(model, test_loader)
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)
                print(
                    f"Epoch {epoch}: Train Loss={epoch_train_loss:.4f}, "
                    f"Train Acc={epoch_train_acc:.4f} | "
                    f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch}: Loss={epoch_train_loss:.4f}, "
                    f"Acc={epoch_train_acc:.4f}"
                )

            # Step the scheduler if provided
            if scheduler is not None:
                # If scheduler is ReduceLROnPlateau, pass validation loss
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if test_loader is not None:
                        scheduler.step(test_loss)
                    else:
                        scheduler.step(epoch_train_loss)
                else:
                    scheduler.step()

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
        Evaluate model on test dataset and compute metrics.

        Args:
            model: Neural network model to evaluate
            test_loader: DataLoader for test data

        Returns:
            tuple: ((loss, accuracy), confusion_matrix)
                - loss: Average test loss
                - accuracy: Test accuracy
                - confusion_matrix: num_classes x num_classes numpy array
        """
        model.to(self.device)
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # For confusion matrix
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)

                # Check for float inputs => multi-label/BCE
                is_multilabel = isinstance(self.criterion, torch.nn.BCEWithLogitsLoss)

                if is_multilabel:
                    labels = labels.to(self.device).float()
                else:
                    labels = labels.to(self.device).long()

                outputs = model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)

                if is_multilabel:
                    preds = (outputs > 0).float()
                    matches = (preds == labels).all(dim=1).float()
                    correct += matches.sum().item()
                    # Confusion matrix is tricky for multi-label, skipping or simple
                    # flatten
                    # Here we just store flattened vectors or skip
                else:
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                total += labels.size(0)

        # Calculate metrics
        avg_loss = total_loss / total
        accuracy = correct / total

        # Build confusion matrix only for single-label
        if len(all_labels) > 0 and not isinstance(
            self.criterion, torch.nn.BCEWithLogitsLoss
        ):
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            num_classes = int(max(all_labels.max(), all_preds.max()) + 1)
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

            for true_label, pred_label in zip(all_labels, all_preds):
                confusion_matrix[true_label, pred_label] += 1
        else:
            # Placeholder for multi-label where simple CM doesn't apply
            confusion_matrix = np.zeros((1, 1), dtype=int)

        return (avg_loss, accuracy), confusion_matrix
