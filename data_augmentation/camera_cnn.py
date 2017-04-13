from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Conv2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, shutil
import theano
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import cv2
import cv2.cv as cv
import math 

# Input image dimensions
img_rows, img_cols =  48, 48

emotions = ['Angry', 'Disgust', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

icons = ['Angry.png', 'Disgust.png', 'Fear.png', 'Happy.png',
           'Sad.png', 'Surprise.png', 'Neutral.png']

icons_path = 'emotions/'

# Load model from json file
json_file = open('model_aug_acc(63).json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into model
loaded_model.load_weights("weight_aug_acc(63).hdf5")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["acc"])

cap = cv2.VideoCapture(0)

while(True):
  # Get the image from the camera
  ret, frame = cap.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  img_resized = cv2.resize(gray, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
  
  # Print the result of prediction of current caputured image
  immatrix = array(img_resized).flatten().reshape(1, img_rows, img_cols, 1)
  result = loaded_model.predict_classes(immatrix/255.0)

  # print emotions[result[0]]
  img_icon = cv2.imread(icons_path + icons[result[0]], 1)
  img_icon_resized = cv2.resize(img_icon, (160, 160))

  # Show the current captured image
  img_show = cv.fromarray(frame)
  cv.ShowImage("Showing Images", img_show)
  cv.ShowImage("Showing Icons", cv.fromarray(img_icon_resized))
  cmd = cv2.waitKey(1) & 0xFF
  if cmd == ord('q'):
    cap.release()
    cv2.destroyAllWindows()
    break