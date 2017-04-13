import numpy as np
import cv2

if __name__ == '__main__':
    print 'Loading data...'
    X_fname = 'data/X_train_total.npy'
    y_fname = 'data/y_train_total.npy'
    X_train = np.load(X_fname)
    y_train = np.load(y_fname)
    print X_train.shape[0]

    print 'saving_data'
    for i in xrange(0, 50, 1):
        file_name = "sample_img/img_" + str(y_train[i]) + "_" + str(i) + ".png"
        cv2.imwrite(file_name, X_train[i])
