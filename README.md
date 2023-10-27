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

# Solution
from google.colab import drive: This line imports the drive module from the google.colab library. This module provides functions for working with Google Drive in Colab.

drive.mount('/content/drive'): This line mounts the Google Drive at the specified directory, which is /content/drive in the Colab notebook. After executing this line, we'll be prompted to authenticate and give permission for Colab to access your Google Drive. Once authorized, The Google Drive will be accessible as if it were a local directory in the Colab environment. After mounted the Google Drive using this code, we can access the files and folders, read and write data, and perform various data analysis or machine learning tasks that involve your Google Drive data within your Colab notebook. It's a convenient way to work with cloud-based data and files in a Colab environment.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/f9f0142f-d762-4ab8-9f30-7d95b29157b7)


Importing necessary libraries: This part of the code imports TensorFlow and various components required for building a deep learning model, including image data preprocessing tools, the VGG16 architecture, layers, models, and the Adam optimizer. Next, it's expected that you would define and configure an image data generator for preprocessing your image data. However, the code you provided is incomplete, and there is no configuration for the data generator.

base_model = VGG16(weights='imagenet', include_top=False) This line loads the VGG16 model pretrained on the ImageNet dataset. The include_top=False argument means that the top classification layers of VGG16 (fully connected layers) are excluded, and you will add your own custom classification layers.

Here, we are adding some custom layers on top of the VGG16 model. The first thing is to apply a global average pooling layer, followed by a dense (fully connected) layer with ReLU activation, and then another dense layer with a softmax activation function. The number of units in the last dense layer (num_classes) should be equal to the number of classes in your classification problem.

model = Model(inputs=base_model.input, outputs=predictions) This line creates a new model that combines the base VGG16 model with your custom classification layers. The input is set to the input of the VGG16 model, and the output is set to your custom predictions.

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) The model, specifying the optimizer (Adam with a learning rate of 0.0001), the loss function (categorical cross-entropy, typically used for classification problems), and the evaluation metric (accuracy)

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/fc173fed-c9dd-4050-a264-bcac4205daa9)

The provided code sets up an ImageDataGenerator object for data augmentation. Data augmentation is a technique commonly used in deep learning for image classification tasks to artificially increase the size of the training dataset by applying various random transformations to the original images. This helps improve the model's generalization and robustness.

rescale=1.0 / 255.0: This parameter rescales the pixel values of the input images. In this case, it divides each pixel value by 255.0, which is a common practice to ensure that pixel values are in the range [0, 1].

rotation_range=20: This parameter specifies the range within which random rotations (in degrees) can be applied to the images. In this case, images can be randomly rotated up to 20 degrees in either direction.

width_shift_range=0.2: This parameter defines the range for random horizontal shifts (as a fraction of the total width of the image). A value of 0.2 means that images can be shifted horizontally by up to 20% of their width in either direction.

height_shift_range=0.2: Similar to width_shift_range, this parameter defines the range for random vertical shifts (as a fraction of the total height of the image). A value of 0.2 means that images can be shifted vertically by up to 20% of their height in either direction.

shear_range=0.2: This parameter controls the shear intensity. Shear transforms slant the shapes of objects in the images. A value of 0.2 means that shear transformations can be applied up to 20%.

zoom_range=0.2: The zoom range specifies the range for random zooming in or out of the images. A value of 0.2 means that images can be zoomed in or out by up to 20%.

horizontal_flip=False: This parameter controls whether random horizontal flips are applied to the images. If set to True, images may be horizontally flipped with a 50% probability.

fill_mode='nearest': The fill mode determines how pixel values are filled when the above transformations cause empty areas in the image. In this case, 'nearest' indicates that the nearest pixels are used to fill the empty regions.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/c3f93395-a4c2-41bc-bf9f-9c3b65261c40)

The provided code is setting up an image data generator and a generator for the training dataset. It appears to be part of a deep learning pipeline for image classification using TensorFlow and Keras. Let's break down the code:

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0): This line creates an ImageDataGenerator object for preprocessing the validation data. It performs pixel value rescaling by dividing each pixel value by 255.0 to ensure that pixel values are in the range [0, 1].

train_generator = train_datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical': This line sets up a generator for the training data by using the flow_from_directory method. Here's what each argument does:

train_dir: This should be a directory containing the training images. The generator will read and preprocess images from this directory.

target_size: This is the size to which the images will be resized during preprocessing. It's typically set to a tuple, e.g., image_size = (224, 224), to ensure that all images are the same size for consistency.

batch_size: This parameter specifies the batch size for training. It determines how many images are processed in each training iteration.

class_mode='categorical': This parameter specifies that the data is used for categorical classification. It means that the generator will expect the images to be organized into subdirectories, where each subdirectory represents a class, and it will generate labels accordingly. This is common for image classification tasks.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/8388ad86-43ff-4468-9d9f-2949adcc46ea)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3)):

This line creates the VGG16 model with the following options:

weights='imagenet':It loads pre-trained weights from the ImageNet dataset, which is a common practice for transfer learning in image classification tasks.

include_top=False: This excludes the top (classification) layer of the VGG16 model, allowing you to add your custom classification layers.

input_shape=(100, 100, 3): It specifies the input shape for your images. In this case, it's set to (100, 100, 3), which means that the model expects input images with a resolution of 100x100 pixels and three color channels (RGB).

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/c757906a-ecca-451f-a7ce-4cbd7061d603)

The provided code constructs a new Keras model by combining the base VGG16 model with the custom classification layers you've defined earlier. Additionally, it freezes (sets as non-trainable) all the layers in the base VGG16 model. Let's break it down step by step:

model = Model(inputs=base_model.input, outputs=predictions):

This line creates a new Keras model (model) by specifying its inputs and outputs. inputs=base_model.input sets the inputs of the new model to be the same as the inputs of the base_model, which is the VGG16 model with custom input shape.

outputs=predictions sets the outputs of the new model to be the predictions generated by the custom layers you defined earlier. These custom layers are connected to the output of the base VGG16 model.

for layer in base_model.layers: layer.trainable = False:

This code iterates through all the layers in the base_model, which is the VGG16 model. For each layer, it sets the trainable attribute to False. This effectively freezes all the layers in the base VGG16 model, preventing them from being updated during the training process.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/b9f84aa3-ea70-4b45-9616-89ea824f4ec3)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy']):

This line compiles the model. It specifies: optimizer=Adam(learning_rate=0.0001): The Adam optimizer with a learning rate of 0.0001 is used to update the model's weights during training. loss='categorical_crossentropy': The categorical cross-entropy loss function, which is commonly used for multi-class classification tasks. metrics=['accuracy']: During training, the model will track and report the classification accuracy as one of the evaluation metrics. epochs = 10: This line sets the number of training epochs. The model will be trained for 10 complete passes through the training dataset.

history = model.fit(...): This is the training loop that fits the model to the training data and validates it on the validation data. The key arguments are as follows:

train_generator: This is the generator for the training dataset, which provides batches of training data during training. steps_per_epoch=train_generator.samples // batch_size: This parameter determines how many steps (batches) are processed per training epoch. It's calculated based on the number of samples in the training dataset and the specified batch size.

validation_data=valid_generator: This is the generator for the validation dataset, which provides batches of validation data for model evaluation during training.

validation_steps=valid_generator.samples // batch_size: Similar to steps_per_epoch, this parameter determines how many steps (batches) are processed per validation epoch, calculated based on the validation dataset size and batch size.

epochs=epochs: This specifies the number of training epochs, as previously defined.

During training, the model's weights are updated using the optimizer (Adam) to minimize the categorical cross-entropy loss. The model's performance is evaluated on both the training and validation datasets, and the training progress (loss and accuracy) is stored in the history variable.

After training for the specified number of epochs, the history object will contain information about the training process, such as the loss and accuracy at each epoch. This information can be used to analyze and visualize how the model's performance evolves over the training process.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/d99773f4-9e1d-4e48-a3b1-6879c9ce2ffa)

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/06060671-f563-49d0-abed-80477278ccdc)

accuracy = model.evaluate(valid_generator)[1]:

model.evaluate(valid_generator) computes the model's loss and metrics on the validation dataset using the validation data generator (valid_generator).

[1] accesses the second element of the list returned by model.evaluate, which corresponds to the accuracy. In the list, the first element is the loss, and the second element is the accuracy.

print("Fine-tuned Model Validation Accuracy: {:.2f}%".format(accuracy * 100)

This line prints the validation accuracy in a human-readable format. It takes the accuracy value (a decimal between 0 and 1) and multiplies it by 100 to convert it to a percentage.

The "{:.2f}%".format(accuracy * 100) part of the code formats the accuracy value to display two decimal places followed by a percentage sign. So, the code calculates the validation accuracy of the fine-tuned model and then prints it as a percentage. This provides an indication of how well the model is performing on the validation dataset after the specified number of training epochs.

![image](https://github.com/Mimran0204/CNN-Implementation/assets/149146008/b5a30307-e7da-441d-a8a5-b82c1b7cb2da)

So, the code calculates the validation accuracy of the fine-tuned model and then prints it as a percentage. This provides an indication of how well the model is performing on the validation dataset after the specified number of training epochs.












