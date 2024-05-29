""" Present simple statistics of this dataset: number of images, their distribution, average
number of white pixels in each class and its standard deviation, the number of common
pixels in each class that are non-white.
"""

import matplotlib.pyplot as pyplot


def run_statitics(dataset):
    result = {
        "number of images": len(dataset),
        "distribution": dataset["label"].value_counts(),
        "average number of white pixels": dataset["white_pixels"].mean(),
        "standard deviation": dataset["white_pixels"].std(),
        "number of common pixels": dataset["white_pixels"].value_counts()[False],
    }
    return result


def plot_image(image):
    pyplot.subplot(330 + 1)
    pyplot.imshow(image, cmap=pyplot.get_cmap("gray"))
    pyplot.show()


# follow the link in https://www.kaggle.com/code/anmolai/mnist-classification-of-digits-accuracy-98
