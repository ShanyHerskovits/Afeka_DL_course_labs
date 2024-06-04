"""
relates to section 8 8. 
Construct a new dataset by generating a new image for each image in the original dataset, where the value of each pixel is replaced by the average of all the surrounding pixels. Repeat steps 5-7 for this dataset.
"""

import numpy as np
import tensorflow as tf
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from plot_utils import save_plot, plot_comparison
from sklearn.decomposition import PCA


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


def create_dataset_with_filter(train, test):
    # Apply the averaging filter to each image in the dataset
    x_train_filtered, x_test_filtered = pixel_surrounding_filter(
        x_train=train[0], x_test=test[0]
    )

    return (x_train_filtered, train[1]), (x_test_filtered, test[1])


# use pca for dimensionality reduction
def apply_pca_reduction(train, test):
    n_components = 50  # Choose the number of components you want to retain
    pca = PCA(n_components=n_components)
    x_train_flattened = train[0].reshape(-1, 28 * 28)
    x_test_flattened = test[0].reshape(-1, 28 * 28)
    x_train_pca = pca.fit_transform(x_train_flattened)
    x_test_pca = pca.transform(x_test_flattened)

    print(x_test_pca.shape)
    return (x_train_pca, train[1]), (x_test_pca, test[1])


# apply 3x3 non-overlapping averaging filter
def rolling_window(image, window_size):
    conv_image = []
    image_height, image_width = image.shape[0], image.shape[1]
    width, height = window_size
    for i in range(0, image_height, height):
        row = []
        for j in range(0, image_width, width):
            row.append(np.mean(image[i : i + height, j : j + width]))
        conv_image.append(np.array(row))

    return np.array(conv_image)


def create_non_overlapping_filter_dataset(train, test):
    # run 3x3 convoultion on non-overlapping frames
    x_train_filtered = np.array([rolling_window(image, (3, 3)) for image in train[0]])
    x_test_filtered = np.array([rolling_window(image, (3, 3)) for image in test[0]])

    # reshape for training and testing later
    return (x_train_filtered.reshape(-1, 10 * 10), train[1]), (
        x_test_filtered.reshape(-1, 10 * 10),
        test[1],
    )
