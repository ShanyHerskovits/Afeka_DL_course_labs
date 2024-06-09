"""
    This file relates to sections 13-14 for imbalance data
"""

from collections import Counter
import numpy as np
import tensorflow as tf
from random import sample
from simple_model import simple_model_main
from datasets_generator import plot_comparison


def prepare_over_and_under_datasets(x, y):
    print("Original class distribution:", Counter(y))
    # Undersample classes 1 and 7
    x_under, y_under = undersample_classes(x, y, 1, 7, n_samples=3500)
    print("Undersample class distribution:", Counter(y_under))
    # Augment classes 3 and 8
    x_over_aug, y_over_aug, augmented_data, augmented_targets, un_augmented_data = (
        augment_classes(x, y, class1=3, class2=8, n_samples=3500)
    )
    print("Oversample class distribution:", Counter(y_over_aug))
    plot_comparison(
        un_augmented_data,
        augmented_data,
        num_images=5,
        filename="augmented_vs_unaugmented",
    )

    train_under = (x_under, y_under)
    train_over = (x_over_aug, y_over_aug)
    return train_under, train_over


# Custom function to add Gaussian noise
def add_gaussian_noise(img, mean=0.0, std=0.1):
    noisy_img = img + np.random.normal(mean, std, img.shape)
    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    return noisy_img


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


# 13. b. Increase the number of images for two classes by performing image manipulations
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
    datagen1 = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=90, horizontal_flip=True
    )
    datagen2 = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=90, vertical_flip=True
    )
    datagen3 = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=90, horizontal_flip=True, vertical_flip=True
    )
    datagen4 = "datagen4"

    augmentation_list = (datagen1, datagen2, datagen3, datagen4)

    # Apply augmentations to each image
    for idx in idx_class1:
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
    un_augmented_data = (
        np.array(un_augmented_data) * 255.0
    )  # Rescale back to original range

    # Create new dataset with augmented data
    new_data = np.concatenate([x, augmented_data])
    new_targets = np.concatenate([y, augmented_targets])

    return new_data, new_targets, augmented_data, augmented_targets, un_augmented_data
