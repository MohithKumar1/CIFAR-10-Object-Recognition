# CIFAR-10-Object-Recognition

This repository demonstrates object recognition on the CIFAR-10 dataset using a deep learning approach. The project includes dataset loading, preprocessing, model training, evaluation, and visualization of predictions.

# Features

Preprocessing and augmenting CIFAR-10 images.
Implementing convolutional neural networks (CNNs) for image classification.
Model training with monitoring and callbacks.
Evaluation of performance using metrics like accuracy and loss.
Visualization of predictions on test images.

Dependencies
Ensure you have the following Python libraries installed:
tensorflow
keras
numpy
matplotlib
seaborn

Dataset

The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. Classes include:

Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck

# Code Overview

1. Dataset Loading and Preprocessing
Load CIFAR-10 dataset using keras.datasets.cifar10.
Normalize pixel values to the range [0, 1].
Split data into training and testing sets.

2. Model Architecture
Build a Convolutional Neural Network (CNN) using keras.Sequential.
Include layers such as Conv2D, MaxPooling2D, Flatten, and Dense.
Use ReLU activation and softmax for output.

3. Model Training
Compile the model with:
Loss: categorical_crossentropy
Optimizer: adam
Metrics: accuracy
Use ModelCheckpoint and EarlyStopping callbacks for efficient training.

4. Model Evaluation
Evaluate the trained model on the test dataset.
Calculate metrics like accuracy and loss.

5. Visualization
Plot training and validation accuracy/loss over epochs.
Display predictions on sample test images.
