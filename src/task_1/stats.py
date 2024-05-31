""" Section 4Present simple statistics of this dataset: number of images, their distribution, average
number of white pixels in each class and its standard deviation, the number of common
pixels in each class that are non-white.
"""

import numpy as np
import tensorflow as tf
from collections import defaultdict
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Combine train and test datasets
x_data = np.concatenate((x_train, x_test), axis=0)
y_data = np.concatenate((y_train, y_test), axis=0)

# Number of images
num_images = x_data.shape[0]
print(f"Number of images: {num_images}")

# Distribution of images across classes
unique, counts = np.unique(y_data, return_counts=True)
distribution = dict(zip(unique, counts))
print(f"Distribution of images across classes: {distribution}")

# Calculate average number of white pixels and standard deviation for each class
avg_white_pixels = {}
std_white_pixels = {}

for i in range(10):
    class_images = x_data[y_data == i]
    white_pixel_counts = (class_images == 0).sum(axis=(1, 2))
    avg_white_pixels[i] = np.mean(white_pixel_counts)
    std_white_pixels[i] = np.std(white_pixel_counts)

print("Average number of white pixels per class:", avg_white_pixels)
print("Standard deviation of white pixels per class:", std_white_pixels)

# Calculate common non-white pixels in each class
common_non_white_pixels = defaultdict(set)

for i in range(10):
    class_images = x_data[y_data == i]
    common_pixels = np.ones_like(class_images[0], dtype=bool)
    for img in class_images:
        common_pixels &= img != 0
    common_non_white_pixels[i] = common_pixels.sum()

print("Number of common non-white pixels per class:", dict(common_non_white_pixels))

# Visualize the average number of white pixels per class
plt.figure(figsize=(10, 5))
plt.bar(range(10), avg_white_pixels.values(), yerr=std_white_pixels.values(), capsize=5)
plt.xlabel("Class")
plt.ylabel("Average number of white pixels")
plt.title("Average number of white pixels per class with standard deviation")
plt.show()


# follow the link in https://www.kaggle.com/code/anmolai/mnist-classification-of-digits-accuracy-98
