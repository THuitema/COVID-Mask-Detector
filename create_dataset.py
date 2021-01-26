import numpy as np
import os
import cv2 as cv
import random
import pickle
import matplotlib.pyplot as plt

DATA_DIR = 'dataset'
CATEGORIES = ['with_mask', 'without_mask']
IMG_SIZE = 100 # training really slows when putting this over 100
face_cascade = cv.CascadeClassifier('haar_cascades/full_face_cascade.xml')

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)  # path to with_mask or without_mask

        # Giving with_mask images a 1 and without_mask images a 0
        if category == 'with_mask':
            category_num = 1
        else:
            category_num = 0

        for img in os.listdir(path):
            try: # in case some image is broken
                img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)  # reads in image & converts to grayscale
                resized_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resizing to make all images the same
                training_data.append([resized_array, category_num])
            except Exception as e:
                pass

create_training_data()


# Shuffling Data
random.shuffle(training_data)

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

# Converting to numpy format
x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # adding grayscale channel at the end
x = x.astype('float32') # converting from uint8
y = np.array(y)

# Normalizing pixel values between 0 and 1
x /= 255

# Saving dataset
pickle_out = open('x.pickle', 'wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

'''
How to open dataset:
pickle_in = open('filename', 'rb')
x = pickle.load(pickle_in)
'''








