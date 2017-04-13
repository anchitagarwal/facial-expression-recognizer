from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.cross_validation import train_test_split
import numpy as np
import time
import sys

def run_cnn(X_train, y_train):
    starttime = time.time()
    X_train = X_train.astype('float32')
    X_train = X_train/255.0
    X_shape = X_train.shape
    y_shape = y_train.shape

    # # Load model from json file
    json_file = open('model_aug(59).json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load weights into model
    model.load_weights("weight_aug_acc(59).hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["acc"])

    confusion_matrix = np.zeros((7, 7), dtype=np.int)
    imgs = X_train
    immatrix = imgs.flatten().reshape(X_shape[0], 48, 48, 1)
    results = model.predict_classes(immatrix)

    print "\n"
    for i in xrange(0, X_shape[0],1):
        predict = results[i]
        label = np.argmax(y_train[i])
        confusion_matrix[predict][label] += 1

    print confusion_matrix

if __name__ == '__main__':
    X_train = np.load('data/X_train_sample_total.npy')
    y_train = np.load('data/y_train_sample_total.npy')

    print "Starting to train cnn"
    run_cnn(X_train, y_train)
