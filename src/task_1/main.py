import tensorflow as tf
from mnist_stats import print_mnist_stats
from simple_model import simple_model_main
from datasets_generator import pixel_surrounding_main

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
# print_mnist_stats(train=train, test=test)

# Section 5,6,7 - please see simple_model.py
# simple_model_main(train=train, test=test)

# Section 8 - please see datasets_generator.py
pixel_surrounding_main(train=train, test=test)
