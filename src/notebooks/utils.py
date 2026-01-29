"""
Visualization utilities.
"""

import matplotlib.pyplot as plt


def plot_loss_and_accuracy(losses, accuracies, test_losses=None, test_accuracies=None):
    """
    Plots training (and optionally test) loss and accuracy on the same figure using two y-axes.

    Args:
        losses (list of float): Training loss values per epoch
        accuracies (list of float): Training accuracy values per epoch
        test_losses (list of float, optional): Test loss values per epoch
        test_accuracies (list of float, optional): Test accuracy values per epoch
    """
    epochs = list(range(1, len(losses) + 1))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Loss on left Y-axis
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    line1 = ax1.plot(
        epochs, losses, color="tab:red", marker="o", label="Train Loss", linestyle="-"
    )[0]
    lines = [line1]
    if test_losses is not None:
        line2 = ax1.plot(
            epochs,
            test_losses,
            color="tab:orange",
            marker="s",
            label="Test Loss",
            linestyle="--",
        )[0]
        lines.append(line2)
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Plot Accuracy on right Y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    line3 = ax2.plot(
        epochs,
        accuracies,
        color="tab:blue",
        marker="x",
        label="Train Accuracy",
        linestyle="-",
    )[0]
    lines.append(line3)
    if test_accuracies is not None:
        line4 = ax2.plot(
            epochs,
            test_accuracies,
            color="tab:cyan",
            marker="^",
            label="Test Accuracy",
            linestyle="--",
        )[0]
        lines.append(line4)
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylim(0, 1.05)  # Accuracy is from 0 to 1

    # Combined legend outside the plot area
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper left",
        bbox_to_anchor=(0, -0.15),
        ncol=4,
        frameon=False,
    )

    # Title and grid
    title = (
        "Training and Test Metrics"
        if test_losses is not None
        else "Training Loss and Accuracy per Epoch"
    )
    plt.title(title)
    fig.tight_layout()
    ax1.grid(True, alpha=0.3)
    plt.show()
