import itertools
import os
import matplotlib.pyplot as plt
import numpy as np


def save_plot(fig, filename, folder="plots"):
    """
    Save a Matplotlib figure to a specified folder.

    Parameters:
    - fig: Matplotlib figure object to be saved.
    - filename: Name of the file to save the figure as (e.g., 'plot.png').
    - folder: Name of the folder to save the figure in (default is 'plots').

    Returns:
    - Full path to the saved figure.
    """
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Construct the full file path
    filepath = os.path.join(folder, filename)

    # Save the figure
    fig.savefig(filepath)

    print(f"Plot saved to {filepath}")
    return filepath


# Plot original and filtered images for comparison
def plot_comparison(original, filtered, num_images=5):
    fig = plt.figure(figsize=(10, 4))
    for i in range(num_images):
        # Original images
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original[i], cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Filtered images
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(filtered[i], cmap="gray")
        plt.title("Filtered")
        plt.axis("off")

    plt.show()
    save_plot(fig, "pixel_surrounding_filter")
    plt.close()


# Plot the training and validation loss
def plot_loss(history, filename):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.gcf()
    plt.show()
    # Save the loss plot
    save_plot(fig, filename)
    plt.close()


# Plot the confusion matrix results
def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues, prefix=""
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize=(10, 8))
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()
    save_plot(fig, prefix + "_confusion_matrix_simple_model")
    plt.close()
