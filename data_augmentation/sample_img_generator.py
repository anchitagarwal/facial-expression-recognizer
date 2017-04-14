import numpy as np
import cv2
from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    print 'Loading data...'
    X_fname = 'data/X_train_sample_total.npy'
    y_fname = 'data/y_train_sample_total.npy'
    X_train = np.load(X_fname)
    y_train = np.load(y_fname)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    print X_train.shape[0]

    print 'saving_data'
    for i in xrange(0, X_train.shape[0], 1):
        file_name = "sample_img/" + classes[np.argmax(y_train[i])] + "/train/" + str(i) + ".png"
        cv2.imwrite(file_name, X_train[i])

    for i in xrange(0, X_test.shape[0], 1):
        file_name = "sample_img/" + classes[np.argmax(y_test[i])] + "/test/" + str(i) + ".png"
        cv2.imwrite(file_name, X_test[i])
