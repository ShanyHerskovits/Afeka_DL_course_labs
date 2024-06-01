"""
relates to section 8 8. 
Construct a new dataset by generating a new image for each image in the original dataset, where the value of each pixel is replaced by the average of all the surrounding pixels. Repeat steps 5-7 for this dataset.
"""

import numpy as np
import tensorflow as tf
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from plot_utils import save_plot


def apply_averaging_filter(image):
    # Define the 3x3 averaging filter
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0

    # Apply convolution using the filter
    filtered_image = convolve(image, kernel, mode="constant", cval=0.0)

    return filtered_image


def pixel_surrounding_filter(x_train, x_test):
    # Apply the averaging filter to each image in the dataset
    x_train_filtered = np.array([apply_averaging_filter(image) for image in x_train])
    x_test_filtered = np.array([apply_averaging_filter(image) for image in x_test])

    # Ensure the new dataset has the same shape
    print(f"Original training data shape: {x_train.shape}")
    print(f"Filtered training data shape: {x_train_filtered.shape}")
    print(f"Original test data shape: {x_test.shape}")
    print(f"Filtered test data shape: {x_test_filtered.shape}")

    return x_train_filtered, x_test_filtered


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


def pixel_surrounding_main(train, test) -> None:
    # Unpack train and test datasets
    x_train, y_train = train
    x_test, y_test = test

    # Apply the averaging filter to each image in the dataset
    x_train_filtered, x_test_filtered = pixel_surrounding_filter(x_train, x_test)

    # Display the comparison
    plot_comparison(x_train, x_train_filtered)
