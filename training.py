import tensorflow as tf
from tensorflow import keras
import sklearn
import numpy as np
import pandas as pd
import kerastuner
from kerastuner.tuners import RandomSearch
import pickle
import matplotlib.pyplot as plt
from config import IMG_SIZE


# highest test accuracy: 99.39% (811/816 test images predicted correctly)


# Loading in data
pickle_in_x = open('x.pickle', 'rb')
x = pickle.load(pickle_in_x)
pickle_in_y = open('y.pickle', 'rb')
y = pickle.load(pickle_in_y)

# Splitting into train and test data
train_images, test_images, train_labels, test_labels = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


model = keras.Sequential([
    keras.layers.AveragePooling2D(6, 3, input_shape=(IMG_SIZE, IMG_SIZE, 1)), # reads a 6x6 square instead of every pixel

    keras.layers.Conv2D(128, 3, activation='relu'), # 300x300x1 color channel (grayscale)
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(64, 3, activation='relu'),

    keras.layers.MaxPool2D(2,2), # Takes the max pixel in every 2,2 grid
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(2, activation='softmax') # outputs: mask (1), no mask (0); softmax is used for classification
])

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(train_images, train_labels, batch_size=16, epochs=12)

model.evaluate(test_images, test_labels)

# Saving model
model.save('./my_model')

# # Just testing the test images again by loading the model
# loaded_model = keras.models.load_model('./my_model')
# loaded_model.evaluate(test_images, test_labels)
#
# predictions = loaded_model.predict(test_images)
#
# num_wrong = 0
# # finds wrong images and displays them
# for x in range(len(predictions)):
#     image = test_images[x].reshape(IMG_SIZE, IMG_SIZE)
#
#     if round(predictions[x][0]) == 1:
#         # plt.imshow(image, cmap='Greys_r')
#         # plt.title(predictions[x])
#         # plt.title('No Mask')
#         if round(predictions[x][1]) != test_labels[x]:
#             print(predictions[x])
#             plt.imshow(image, cmap='Greys_r')
#             plt.title('Predicted: No Mask, Actual: Mask')
#             num_wrong += 1
#     elif round(predictions[x][1]) == 1:
#         # plt.imshow(image, cmap='Greys_r')
#         # plt.title(predictions[x])
#         # plt.title('Mask')
#         if round(predictions[x][1]) != test_labels[x]:
#             print(predictions[x])
#             plt.imshow(image, cmap='Greys_r')
#             plt.title('Predicted: Mask, Actual: No Mask')
#             num_wrong += 1
#     plt.show()
#
#
# print('Correct: ' + str(len(test_labels) - num_wrong) + '/' + str(len(test_labels)))
#
#




