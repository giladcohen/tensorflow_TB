'''This code converts a numpy image to .bin in the same format of cifar10'''
from __future__ import division


import numpy as np
from keras.datasets import cifar10
import cv2
import os

def convert_numpy_to_bin(images, labels, save_file, h=32, w=32):
    images = (np.array(images))
    N = images.shape[0]
    record_bytes = 3 * h * w + 1 #includes also the label
    out = np.zeros([record_bytes * N], np.uint8)
    for i in range(N):
        im = images[i]
        r = im[:,:,0].flatten()
        g = im[:,:,1].flatten()
        b = im[:,:,2].flatten()
        label = labels[i]
        out[i*record_bytes:(i+1)*record_bytes] = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
    out.tofile(save_file)

def save_cifar10_to_disk(train_data_dir, train_labels_file, test_data_dir, test_labels_file):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    np.savetxt(train_labels_file, Y_train, fmt='%0d')
    np.savetxt(test_labels_file,  Y_test,  fmt='%0d')
    for i in range(X_train.shape[0]):
        img = X_train[i]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(train_data_dir, 'train_image_%0d.png' % i), img_bgr)
    for i in range(X_test.shape[0]):
        img = X_test[i]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(test_data_dir,  'test_image_%0d.png'  % i), img_bgr)
