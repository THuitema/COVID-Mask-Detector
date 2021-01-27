import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from config import IMG_SIZE
from tensorflow import keras

img = cv.imread('dataset/test01.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Neural network model
loaded_model = keras.models.load_model('./my_model')

# Haar Cascades
full_face_cascade = cv.CascadeClassifier('haar_cascades/full_face_cascade.xml')
profile_face_cascade = cv.CascadeClassifier('haar_cascades/profile_face_cascade.xml')

image_list = []

def detect(img, cascade1, cascade2):
    full_face_rectangles = cascade1.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    profile_face_rectangles = cascade2.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    # if it detects a full face
    if len(full_face_rectangles) > 0:
        return full_face_rectangles

    # if it detects a profile face
    elif len(profile_face_rectangles) > 0:
        return profile_face_rectangles

    # if no faces exist, it returns nothing
    else:
        return []

def get_face_data(rects):
    predict_data = []

    # running through detected faces
    for (x, y, w, h) in rects:
        # making array of face pixels
        face_array = np.array(gray[y:y + h, x:x + w])  # putting grayscale pixel values from face
        resized_array = cv.resize(face_array, (IMG_SIZE, IMG_SIZE)) # making it the same size as training data
        predict_data.append([resized_array])

    # moving to numpy format
    predict_data = np.array(predict_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    predict_data = predict_data.astype('float32')
    predict_data /= 255 # normalizing data

    return predict_data


def predict(data):
    predictions = loaded_model.predict(data)
    return predictions


def draw_rects(img, rects, predictions):
    print(predictions)
    for counter, (x, y, w, h) in enumerate(rects):

        # no mask predicted
        if round(predictions[counter][0]) == 1:
            # rectangle from face_detect shown w/ text showing prediction %
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            cv.putText(img, ('No Mask ' + str(round(predictions[counter][0] * 100, 2)) + '%'), (x, y-10), cv.FONT_HERSHEY_DUPLEX, 0.55, (0,0,255), thickness=1)

        # mask predicted
        elif round(predictions[counter][1]) == 1:
            # rectangle from face_detect shown w/ text showing prediction %
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv.putText(img, ('Mask: ' + str(round(predictions[counter][1] * 100, 2)) + '%'), (x, y-10), cv.FONT_HERSHEY_DUPLEX, 0.55, (0,255,0), thickness=1)


rects = detect(gray, full_face_cascade, profile_face_cascade)
predict_data = get_face_data(rects)
predictions = predict(predict_data)
draw_rects(img, rects, predictions)

cv.imshow('Face Detect', img)
cv.waitKey(0)

