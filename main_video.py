import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from keras.models import load_model
from scipy.spatial import distance
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
webcam = cv2.VideoCapture(0)  # Use camera 0
MIN_DISTANCE = 100
face_model = cv2.CascadeClassifier('input/haarcascades/haarcascade_frontalface_default.xml')
model = load_model('masknet.h5')
mask_label = {0: 'MASK', 1: 'NO MASK'}
dist_label = {0: (0, 255, 0), 1: (255, 0, 0)}
while True:
    rval, img = webcam.read()
    img = cv2.flip(img, 1, 1)  # Flip to act as a mirror



    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)  # returns a list of (x,y,w,h) tuples
    label = [0 for i in range(len(faces))]
    for i in range(len(faces) - 1):
        for j in range(i + 1, len(faces)):
            dist = distance.euclidean(faces[i][:2], faces[j][:2])
            if dist < MIN_DISTANCE:
                label[i] = 1
                label[j] = 1
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # colored output image
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        crop = new_img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (128, 128))
        crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
        mask_result = model.predict(crop)
        cv2.putText(new_img, mask_label[mask_result.argmax()], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    dist_label[label[i]], 2)
        cv2.rectangle(new_img, (x, y), (x + w, y + h), dist_label[label[i]], 1)

    img=new_img
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Show the image
    cv2.imshow('LIVE', img)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27:  # The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()