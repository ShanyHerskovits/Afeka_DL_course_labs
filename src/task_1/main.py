from collections import Counter

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from random import sample

from mnist_stats import print_mnist_stats
from simple_model import simple_model_main
from datasets_generator import pixel_surrounding_main, plot_comparison

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
    return train, test, x_train, y_train

# This is the original dataset we are going to use
train, test, X, y = load_mnist()

# Section 4 - please see mnist_stats.py
# print_mnist_stats(train=train, test=test)

# Section 5,6,7 - please see simple_model.py
# simple_model_main(train=train, test=test)

# Section 8 - please see datasets_generator.py
# pixel_surrounding_main(train=train, test=test)

# section 13

print('Original class distribution:', Counter(y))


# a. Perform undersampling for two classes within the existing 10 classes


def undersample_classes(x, y, class1, class2, n_samples):
    # Get indices of the two classes
    idx_class1 = np.where(y == class1)[0]
    idx_class2 = np.where(y == class2)[0]

    # Randomly sample n_samples from each class
    idx_class1 = np.random.choice(idx_class1, n_samples, replace=False)
    idx_class2 = np.random.choice(idx_class2, n_samples, replace=False)

    # Create new dataset with undersampled classes
    idx_other_classes = np.where((y != class1) & (y != class2))[0]
    new_indices = np.concatenate([idx_class1, idx_class2, idx_other_classes])

    return x[new_indices], y[new_indices]


# Undersample classes 1 and 7
x_under, y_under = undersample_classes(X, y, 1, 7, n_samples=3500)
print('Undersample class distribution:', Counter(y_under))


# Custom function to add Gaussian noise
def add_gaussian_noise(img, mean=0.0, std=0.1):
    noisy_img = img + np.random.normal(mean, std, img.shape)
    noisy_img = np.clip(noisy_img, 0., 1.)
    return noisy_img


# 13. b. Increase the number of images for two classes by performing image manipulations

# Helper function to visualize a sample image
def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def augment_classes(x, y, class1, class2, n_samples=500):
    # Get indices of the two classes
    idx_class1 = np.where(y == class1)[0]
    idx_class2 = np.where(y == class2)[0]

    # Randomly sample n_samples from each class
    idx_class1 = np.random.choice(idx_class1, n_samples, replace=False)
    idx_class2 = np.random.choice(idx_class2, n_samples, replace=False)

    augmented_data = []
    augmented_targets = []
    un_augmented_data = []

    # Define augmentation transformations
    datagen1 = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90, horizontal_flip=True)
    datagen2 = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90, vertical_flip=True)
    datagen3 = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90, horizontal_flip=True, vertical_flip=True)
    datagen4 = 'datagen4'

    augmentation_list = (datagen1, datagen2, datagen3, datagen4)

    # Apply augmentations to each image
    for idx in idx_class1:
        un_augmented_data.append(x[idx])
        img = x[idx] / 255.0  # Normalize image
        img = img.reshape((1,) + img.shape + (1,))
        for dgen in sample(augmentation_list, 1):
            if dgen == datagen4:
                augmented_img = add_gaussian_noise(img)
                augmented_img = augmented_img.reshape(28,28)
            else:
                 for batch in dgen.flow(img, batch_size=1):
                     augmented_img = batch[0].reshape(28, 28)
                     break
            augmented_data.append(augmented_img)
            augmented_targets.append(class1)


    for idx in idx_class2:
        un_augmented_data.append(x[idx])
        img = x[idx] / 255.0  # Normalize image
        img = img.reshape((1,) + img.shape + (1,))
        for dgen in sample(augmentation_list, 1):
            if dgen == datagen4:
                augmented_img = add_gaussian_noise(img)
                augmented_img = augmented_img.reshape(28, 28)
            else:
                 for batch in dgen.flow(img, batch_size=1):
                     augmented_img = batch[0].reshape(28, 28)
                     break
            augmented_data.append(augmented_img)
            augmented_targets.append(class2)


    # Convert augmented data to numpy arrays
    augmented_data = np.array(augmented_data) * 255.0  # Rescale back to original range
    un_augmented_data = np.array(un_augmented_data) * 255.0  # Rescale back to original range

    # Create new dataset with augmented data
    new_data = np.concatenate([x, augmented_data])
    new_targets = np.concatenate([y, augmented_targets])

    return new_data, new_targets, augmented_data, augmented_targets, un_augmented_data


# Augment classes 3 and 8
x_over_aug, y_over_aug, augmented_data, augmented_targets, un_augmented_data = augment_classes(X, y, class1=3, class2=8, n_samples=3500)
print('Oversample class distribution:', Counter(y_over_aug))


plot_comparison(un_augmented_data, augmented_data, num_images=5)

train_under = x_under, y_under
train_over_aug = x_over_aug, y_over_aug
# Section 14 data from 13 a - undersampling of two classes  - (Sections 11 -> 5,6,7) please see simple_model.py
simple_model_main(train=train_under, test=test)
# Section 14 data from 13 b - Oversampling of two classes by image augmentation  - (Sections 11 -> 5,6,7) please see simple_model.py
simple_model_main(train=train_over_aug, test=test)

