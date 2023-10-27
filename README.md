# CNN-Implementation
# Task 1
Implement a basic CNN model from scratch using TensorFlow for image classification. In our dataset, we have 3 classes of digits which are 0, 1 and 2. What you need to do is: 1. Split the dataset into training and validation set. 80% for training and the remaining 20% for validation. You should do this split in code. You are only provided with a single folder which is having all the 3 classes. Each sub-folder in the Hand_written_digits folder is considered a separate class/category by the TensorFlow. TensorFlow allows you to do the split. Check TensorFlow official documentation to see how it is done. Images in the dataset are gray scale of size (100x100) pixel. The dataset provided to you is in the following format:

                  Hand_written_digits

                        0_digits

                        1_digits

                        2_digits
Train model on the provided dataset. Do the data augmentation suitable for the dataset provided. Be careful while doing the augmentation as all the augmentations are not suitable for our dataset like doing the vertical flip or horizontal flip on digit 2 changes it to something else. Train the model on the provided hand-written digits dataset and report the accuracy achieved after 10 epochs of training. You can increase the number of epochs as per your choice but be careful of model overfitting. This task will be marked based on validation accuracy so do play around with the model architecture to achieve the best possible accuracy. Following is the model architecture. You can change the model architecture as per your creativity.
# Solution
from google.colab import drive: This line imports the drive module from the google.colab library. This module provides functions for working with Google Drive in Colab.

drive.mount('/content/drive'): This line mounts the Google Drive at the specified directory, which is /content/drive in the Colab notebook. After executing this line, we'll be prompted to authenticate and give permission for Colab to access your Google Drive. Once authorized, The Google Drive will be accessible as if it were a local directory in the Colab environment.

After mounted the Google Drive using this code, we can access the files and folders, read and write data, and perform various data analysis or machine learning tasks that involve your Google Drive data within your Colab notebook. It's a convenient way to work with cloud-based data and files in a Colab environment.
