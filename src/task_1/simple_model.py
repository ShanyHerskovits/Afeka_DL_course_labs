"""
    Apply a simple neural network to this dataset, in order to perform classification. (You can
    use the `tensorflowlib.py` program from Moodle). Describe the results of different
    measures: accuracy, precision, recall, F1, sensitivity, and specificity.
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
from plot_utils import save_plot


# preprocess data
def preprocess(train, test):
    # Unpack train and test datasets
    x_train, y_train = train
    x_test, y_test = test

    # Flatten the images for the neural network input
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    return x_train, y_train, x_test, y_test


def build_model(input_shape):
    # Build the model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, activation="relu", input_shape=input_shape),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    # Train the model
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

    # plot trainig loss - section 7 in the task
    plot_loss(history)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test accuracy:", test_acc)

    # Predict the labels for the test data
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Generate the classification report
    report = classification_report(y_test, y_pred_classes, output_dict=True)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    evaluate_results(report, conf_matrix)
    plot_confusion_matrix(conf_matrix, classes=range(10), title="Confusion Matrix")


def evaluate_results(report, conf_matrix):
    # Extract precision, recall, F1-score, and support for each class
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1_score = report["weighted avg"]["f1-score"]
    support = report["weighted avg"]["support"]

    # Sensitivity (Recall) and Specificity
    sensitivity = recall
    specificity = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

    print(f"Precision: {precision}")
    print(f"Recall (Sensitivity): {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")


# Section 6
def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
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
    save_plot(fig, "confusion_matrix_simple_model")
    plt.close()


# Plot the training and validation loss
def plot_loss(history):
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
    save_plot(fig, "simple_model_training_loss.png")
    plt.close()


def simple_model_main(train, test):
    # Preprocess the data
    x_train, y_train, x_test, y_test = preprocess(train, test)

    # Build the model
    input_shape = (28 * 28,)
    model = build_model(input_shape=input_shape)

    # Train and evaluate the model
    train_and_evaluate_model(model, x_train, y_train, x_test, y_test)
