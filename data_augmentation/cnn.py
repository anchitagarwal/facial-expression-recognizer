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

    # data augmentation:
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    datagen = ImageDataGenerator(rotation_range=10,
                                 shear_range=0.15,
                                 width_shift_range=0.15,
                                 height_shift_range=0.15,
                                 horizontal_flip=True)

    datagen.fit(X_train)

    # model = Sequential()

    # model.add(Conv2D(24, 3, input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    # model.add(Conv2D(48, 3))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    # model.add(Conv2D(144, 3))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    # model.add(Flatten())

    # model.add(Dense(500))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    # model.add(Dense(500))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    # prediction = model.add(Dense(y_train.shape[1], activation='softmax'))
    # optimizer:
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])    

    # # Load model from json file
    json_file = open('model_aug_acc(63).json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load weights into model
    model.load_weights("weight_aug_acc(63).hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["acc"])

    print 'Training....'
    total_era = 100;
    while(total_era > 0):
        hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=512),
            samples_per_epoch=len(X_train), nb_epoch=10,verbose=1)

        # Save samples of data_augmentation to preview folder. Use x_train_1pct as training data set to avoid generate too many samples 
        # hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=28, save_to_dir='preview', save_prefix='face_aug', save_format='jpeg'),
        #     samples_per_epoch=len(X_train), nb_epoch=10,verbose=1)

        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        model_json = model.to_json()
        with open("model_aug(" + str(int(score[1]*100)) +").json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("weight_aug_acc(" + str(int(score[1]*100)) +").hdf5", overwrite=True)

        total_era -= 1
    return

if __name__ == '__main__':
    X_train = np.load('data/X_train_sample_total.npy')
    y_train = np.load('data/y_train_sample_total.npy')

    print "Starting to train cnn"
    run_cnn(X_train, y_train)
