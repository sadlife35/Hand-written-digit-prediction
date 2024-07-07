Title of Project:
Handwritten Digit Classification using MNIST Dataset 
Objective:
To build a machine learning model that can accurately classify handwritten digits (0-9) from the MNIST dataset using Convolutional Neural Networks (CNN).
Data Source:
The MNIST dataset, which contains 70,000 images of handwritten digits. The dataset is available from TensorFlow/Keras datasets.
Import Library:
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
Import Data:
# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
Describe Data:
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Output:
# Training data shape: (60000, 28, 28)
# Training labels shape: (60000,)
# Test data shape: (10000, 28, 28)
# Test labels shape: (10000,)
Data Visualization:
# Visualize some of the training data
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='binary')
    ax.set(title=f"Label: {y_train[i]}")
    ax.axis('off')
plt.show()
Data Preprocessing:
# Reshape and normalize the images
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
Define Target Variable (y) and Feature Variables (X):
# Feature variables (X) are the images
X = X_train

# Target variable (y) are the labels
y = y_train
Define Target Variable (y) and Feature Variables (X):
Train Test Split:
The MNIST dataset is already split into training and test sets by default. If needed, we can split the training set further into training and validation sets, but for simplicity, we will use the provided split.
Modeling:# Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Output layer for 10 classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
Model Evaluation:
# Evaluate the model on the test data
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
Prediction:
# Make predictions on the test data
predictions = model.predict(X_test)

# Plot some predictions
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='binary')
    ax.set(title=f"Pred: {predictions[i].argmax()}\nTrue: {y_test[i].argmax()}")
    ax.axis('off')
plt.show()
Explanation:
This project demonstrates the classification analysis of handwritten digits using the MNIST dataset. The MNIST dataset is a standard benchmark in machine learning for image processing tasks. We employed a Convolutional Neural Network (CNN) due to its effectiveness in handling image data. The data was preprocessed by normalizing and reshaping the images and one-hot encoding the labels. The model was trained on the training set and evaluated on the test set, achieving high accuracy. Predictions were visualized to compare the model's output with the true labels, showcasing the model's performance.
