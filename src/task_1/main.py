import tensorflow as tf
import numpy as np
from mnist_stats import print_mnist_stats
from simple_model import simple_model_main
from datasets_generator import (
    apply_pca_reduction,
    create_dataset_with_filter,
    create_non_overlapping_filter_dataset,
    pixel_surrounding_filter,
)
from plot_utils import plot_comparison

# 1. Choose a Python environment and install it on your computer (PyCharm or Google Colab).
print(
    "We have used virtual environment and install all requirements listed in requirements.txt"
)

# 2. Install the following libraries: numpy, matplotlib, pandas, tensorflow, sklearn, and pytorch.
# Please see file requirements.txt for details.


# section 3 Write a program that loads the MNIST dataset.
def load_mnist():
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train = (x_train, y_train)
    test = (x_test, y_test)
    return train, test


# This is the original dataset we are going to use
train, test = load_mnist()

# Section 4 - please see mnist_stats.py
print_mnist_stats(train=train, test=test)

# Section 5,6,7 - please see simple_model.py
simple_model_main(train=train, test=test, file_prefix="original")

# Section 8 - please see datasets_generator.py
train_filtered, test_filtered = create_dataset_with_filter(train=train, test=test)
plot_comparison(original=train[0], filtered=train_filtered[0], num_images=5)

# re run 5-7 steps for the data after the filter
simple_model_main(train=train_filtered, test=test_filtered, file_prefix="filtered_avg")

# section 10 a
train_pca, test_pca = apply_pca_reduction(train, test)

# running model on pca data
simple_model_main(train=train_pca, test=test_pca, file_prefix="pca", input_shape=(50,))

# section 10 b
train_non_ovelapping, test_non_ovelapping = create_non_overlapping_filter_dataset(
    train, test
)

# section 11 - running model on 3x3 convolution non overlapping
simple_model_main(
    train=train_non_ovelapping,
    test=test_non_ovelapping,
    file_prefix="non_overlapping",
    input_shape=(10 * 10,),
)
