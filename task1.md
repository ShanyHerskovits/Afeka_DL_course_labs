DataScience2 lab
Dr. Sharon Yalov-Handzel
Homework 1
You can submit homework as a _.py_ or _.ipynb_ file. Additionally, you can write the theoretical answers in a Word document.
1. Choose a Python environment and install it on your computer (PyCharm or Google Colab). 
2. Install the following libraries: numpy, matplotlib, pandas, tensorflow, sklearn, and pytorch. 
3. Write a program that loads the MNIST dataset.
4. Present simple statistics of this dataset: number of images, their distribution, average number of white pixels in each class and its standard deviation, the number of common pixels in each class that are non-white.
5. Apply a simple neural network to this dataset, in order to perform classification. (You can use the `tensorflowlib.py` program from Moodle). Describe the results of different measures: accuracy, precision, recall, F1, sensitivity, and specificity.
6. Describe the confusion matrix of the above classification. What conclusions can be drawn from this matrix?
7. Show a figure of the Loss along the training. What is the optimal number of epochs?
8. Construct a new dataset by generating a new image for each image in the original dataset, where the value of each pixel is replaced by the average of all the surrounding pixels. Repeat steps 5-7 for this dataset.
9. Discuss the comparison between the results of the classification applied to the original dataset and the modified dataset.
10. Generate two new datasets with reduced dimensions:
a. Perform dimension reduction by Principal Component Analysis (PCA).
b. Perform dimension reduction by replacing each non-overlapping 3x3 pixel block with its average value.
11. Apply steps 5-7 to each of these new datasets.
12. Compare the results of the classification applied to each of the reduced datasets and the original dataset.
13. Generate two new imbalanced datasets derived from the original dataset:
a. Perform undersampling for two classes within the existing 10 classes.
b. Choose two classes among the existing 10, and increase the number of images belonging to these classes by performing image manipulations like rotation, flipping, blurring, etc.
14. Repeat steps 11-12 for these new datasets.