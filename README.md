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

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/824638d1-98bd-430d-93d2-43cdf811564f)
![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/e671b27d-2774-4bb9-9974-ca4eb72427d5)

import tensorflow as tf: 
This imports the TensorFlow library, which is a popular open-source machine learning framework for building and training neural networks.

from tensorflow.keras.preprocessing.image import ImageDataGenerator: 
This imports the ImageDataGenerator class from the tensorflow.keras.preprocessing.image module. ImageDataGenerator is a tool for data augmentation and preprocessing when working with image data, which is commonly used when training neural networks on images.

from tensorflow.keras.models import Sequential: 
This imports the Sequential class from tensorflow.keras.models. The Sequential model is used for building neural networks layer by layer in a linear or sequential manner.

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense: 
This imports specific layer types commonly used in neural network architectures.

Conv2D is a 2D convolutional layer used for feature extraction in convolutional neural networks (CNNs).

MaxPooling2D is a 2D pooling layer used for down-sampling in CNNs.

Flatten is a layer that flattens the output of previous layers to prepare it for fully connected layers.

Dense is a fully connected layer, also known as a dense layer, used for classification or regression tasks.

from sklearn.model_selection import train_test_split: 
This imports the train_test_split function from the sklearn.model_selection module. This function is used to split your dataset into training and testing sets, which is a common step when building and evaluating machine learning models.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/c57c97c6-aad9-4bae-9c61-fdc1c09a496a)

train_dir and valid_dir: 
These variables specify the directories containing your training and validation datasets, respectively. In this case, both are set to 'hand_written_digits'. It's assumed that you have your image data organized within these directories, and you'll use these directories to generate training and validation data.
image_size: 
This variable specifies the target size for your images. In this case, the images will be resized to 100x100 pixels.
batch_size:
This variable determines the batch size for your data generator. The batch size is the number of images processed in each iteration during training.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/397700ed-8f17-4c5f-b31e-6024c3e7a257)

train_datagen and valid_datagen: These are instances of ImageDataGenerator from Keras. These generators are used for data augmentation and preprocessing. Here's a breakdown of the data augmentation options applied in the train_datagen:

rescale: Images are rescaled so that their pixel values are in the range [0, 1].

rotation_range: Randomly rotates the images by up to 20 degrees.

width_shift_range and height_shift_range: Randomly shifts the width and height of the images by up to 20%.

shear_range: Randomly applies shear transformation.

zoom_range: Randomly zooms in on the images by up to 20%.

horizontal_flip: Randomly flips the images horizontally (left to right).

fill_mode: Determines how the generator fills in newly created pixels after applying transformations. In this case, it's set to 'nearest,' which means it will fill with the nearest available pixel value.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/83425a80-cd5f-4b3d-9d2c-04e27522770f)

train_generator and valid_generator: These are data generators created using ImageDataGenerator to load and preprocess images from the training and validation directories. They are set up with the following parameters:

flow_from_directory: This method reads and preprocesses images from the specified directory.

train_dir and valid_dir: The directories from which images are loaded.

target_size: The target image size is set to (100, 100).

batch_size: The batch size is set to 32.

class_mode:The class_mode is set to 'categorical,' which implies that this code is likely designed for a multi-class classification problem, where images are associated with one of multiple classes.

These data generators are typically used in deep learning workflows for image classification tasks. They handle the loading, preprocessing, and batching of the training and validation data, making it easier to train deep learning models on large image datasets. The data augmentation applied in the training data generator helps increase model robustness by exposing it to various forms of the same image, which can improve generalization.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/dbd93339-1c41-4892-8fa3-b674aead509d)

model = Sequential(): This line initializes a sequential model. A sequential model is a linear stack of layers, and you can add layers to it one by one in a sequential manner.

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)): This adds the first convolutional layer to the model. Here's what each argument does:

32 is the number of filters (output channels) in this convolutional layer.

(3, 3) specifies the size of the convolutional kernel (3x3).

activation='relu' sets the activation function for this layer to the Rectified Linear Unit (ReLU).

input_shape=(100, 100, 3) defines the shape of the input data. It expects 100x100 pixel RGB images (3 color channels).

model.add(MaxPooling2D((2, 2)): This adds a max-pooling layer after the first convolutional layer. Max-pooling reduces the spatial dimensions of the feature maps, which helps in reducing the number of parameters and computation.

The next two blocks of code (model.add(Conv2D(64, (3, 3), activation='relu') followed by model.add(MaxPooling2D((2, 2)) and model.add(Conv2D(128, (3, 3), activation='relu') followed by model.add(MaxPooling2D((2, 2))) add additional pairs of convolutional and max-pooling layers. These create deeper and more complex feature representations as you progress through the network.

model.add(Flatten()): This layer flattens the 2D feature maps into a 1D vector. It's a necessary step before connecting to fully connected layers.

model.add(Dense(128, activation='relu'): This adds a fully connected layer with 128 units and ReLU activation. This layer is responsible for learning higher-level features from the flattened feature vectors.

model.add(Dense(3, activation='softmax'): This adds the output layer with 3 units and a softmax activation function. It's likely that this model is designed for a classification task with three classes. The softmax activation will provide probability distributions over these classes, and the model will make predictions based on the class with the highest probability.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/27c38ffb-a308-4e85-9e62-de032ca31367)

optimizer='adam': This argument specifies the optimizer to be used during training. In this case, it's set to 'adam,' which is a popular and widely used optimization algorithm in deep learning. The Adam optimizer is known for its efficiency and effectiveness in training neural networks.

loss='categorical_crossentropy': This argument specifies the loss function to be used during training. 'Categorical cross-entropy' (or 'categorical_crossentropy') is a common choice for multi-class classification problems, where the model is expected to predict one of several possible classes. This loss function quantifies the error between the predicted class probabilities and the actual class labels.

metrics=['accuracy']: This argument specifies the evaluation metric(s) to be used during training and evaluation. In this case, it's set to 'accuracy,' which is a standard metric for classification tasks. It measures the proportion of correctly classified examples in the validation dataset.

When we compile the model using this code, we are configuring it for training. The model will use the 'adam' optimizer to minimize the 'categorical_crossentropy' loss, and during training, it will keep track of the 'accuracy' metric to assess how well the model is performing on the training and validation data.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/19c42331-850a-47eb-a3f7-340bb62248c0)

The provided code is used to train a deep learning model (model) on your dataset using a training generator (train_generator) and a validation generator (valid_generator). Let's break down what each part of the code does:

epochs = 10: This line sets the number of training epochs. An epoch is one complete pass through the entire training dataset. In this case, the model will be trained for 10 epochs, meaning it will go through the training dataset 10 times, updating its weights to minimize the specified loss.

history = model.fit(...): This line invokes the fit method of the model to start the training process. The fit method trains the model using the specified data generators and training parameters. Here's what each argument does:

train_generator: This is the training data generator, which provides batches of training data to the model.

steps_per_epoch: This parameter determines how many batches are processed in each epoch. It's set to train_generator.samples // batch_size, which ensures that the model goes through the entire training dataset once per epoch.

validation_data: This is the validation data generator, used to evaluate the model's performance on a separate dataset during training.

validation_steps: Similar to steps_per_epoch, this parameter determines how many batches are processed in each validation epoch. It's set to valid_generator.samples // batch_size.

epochs: The number of training epochs, which is set to 10 in this case.

After executing this code, the model will go through 10 epochs of training, during which it will update its weights using the training data and evaluate its performance on the validation data. The training and validation metrics (loss and accuracy) for each epoch will be recorded in the history object, which we can use to visualize and analyze the training progress. The model will be gradually optimized to make better predictions on the data, assuming that the architecture and data are suitable for the task.


![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/a3a44bf2-9f95-436d-91ca-8912e37e735e)
![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/49da36b6-03e3-4530-b0c8-eeb8632917ea)

The provided code is used to evaluate the trained deep learning model on a validation dataset and then print the validation accuracy. Here's what each part of the code does:

accuracy = model.evaluate(valid_generator)[1]: This line evaluates the model's performance on the validation dataset using the evaluate method. Specifically, it calculates the accuracy of the model's predictions on the validation data. The [1] index is used to extract the accuracy from the evaluation results.

print("Validation Accuracy: {:.2f}%".format(accuracy * 100)):This line prints the validation accuracy as a percentage. It formats the accuracy value to have two decimal places and displays it as a percentage.

So, when we run this code, it will calculate and print the validation accuracy of your trained model. This is a common practice in machine learning and deep learning to assess how well the model is performing on unseen data. The accuracy represents the proportion of correctly predicted samples in the validation dataset, expressed as a percentage.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/59e3deda-3342-498d-880f-dd3fb34b2d0b)


# Task 2
Now this time do Task 1 again by fine-tuning the pre-trained model of your choice. Contents of week 6 and week 7 are focused on the concept of transfer learning so the main objective of this task is to solidify and reinforce your core concepts of transfer learning and how to fine-tune a pre-trained model trained on different dataset for your own custom dataset. Observe the accuracy of the model and compare it with the accuracy of the model in task 1.




