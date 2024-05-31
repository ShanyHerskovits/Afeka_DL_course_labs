import tensorflow as tf


## section 3 Write a program that loads the MNIST dataset.
def load_mnist():
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    return (x_train, y_train), (x_test, y_test)


# Section 4 - please see stats.py

# Section 5 - please see simple_model.py
