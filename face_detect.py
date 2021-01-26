import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0)

# Dimensions
camera.set(3, 1280)
camera.set(4, 720)

# Haar Cascades
full_face_cascade = cv.CascadeClassifier('haar_cascades/full_face_cascade.xml')
profile_face_cascade = cv.CascadeClassifier('haar_cascades/profile_face_cascade.xml')

def detect(img, cascade1, cascade2):
    full_face_rectangles = cascade1.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100))
    profile_face_rectangles = cascade1.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100))

    # if it detects a full face
    if len(full_face_rectangles) > 0:
        return full_face_rectangles, (0, 255, 0)

    # if it detects a profile face
    elif len(profile_face_rectangles) > 0:
        return profile_face_rectangles, (0, 0, 255)

    # if no faces exist, it returns nothing
    else:
        return [], ()


def draw_rects(img, rects, color):
    for (x, y, w, h) in rects:
        cv.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
        cv.rectangle(img, (x, y), (x + w, y + 20), color, thickness=-1)
        cv.putText(img, 'Face', (x, y + 16), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), thickness=2)

while True:
    ret, frame = camera.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detecting and drawing rectangles
    rectangles, rect_color = detect(frame, full_face_cascade, profile_face_cascade)
    draw_rects(frame, rectangles, rect_color)
    cv.imshow('Detect', frame)

    # Press q to break
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()