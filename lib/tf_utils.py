'''This code converts a numpy image to .bin in the same format of cifar10'''

import numpy as np
from keras.datasets import cifar10
import cv2
import os
import tensorflow as tf


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

class LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def __init__(self, hps, model, TRAIN_BATCH_SIZE, cap):
        self.hps = hps
        self.model = model
        self.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
        self.cap = cap
        self.setter_done = False
        self.notify = [False, False, False, False]

    def begin(self):
        self._lrn_rate = self.hps.lrn_rate

    def before_run(self, run_context):
        if self.setter_done:
            return tf.train.SessionRunArgs(
                self.model.global_step,  # Asks for global step value.
                feed_dict={self.model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
        if self.setter_done:
            train_step = run_values.results
            epoch = (self.TRAIN_BATCH_SIZE * train_step) // self.cap
            if epoch < 60:
                self._lrn_rate = self.hps.lrn_rate
                if not self.notify[0]:
                    tf.logging.info('epoch=%0d. Decreasing learning rate to %.8f' %(epoch, self._lrn_rate))
                    self.notify[0] = True
            elif epoch < 120:
                self._lrn_rate = self.hps.lrn_rate/5
                if not self.notify[1]:
                    tf.logging.info('epoch=%0d. Decreasing learning rate to %.8f' %(epoch, self._lrn_rate))
                    self.notify[1] = True
            elif epoch < 160:
                self._lrn_rate = self.hps.lrn_rate/25
                if not self.notify[2]:
                    tf.logging.info('epoch=%0d. Decreasing learning rate to %.8f' %(epoch, self._lrn_rate))
                    self.notify[2] = True
            else:
                self._lrn_rate = self.hps.lrn_rate/125
                if not self.notify[3]:
                    tf.logging.info('epoch=%0d. Decreasing learning rate to %.8f' %(epoch, self._lrn_rate))
                    self.notify[3] = True

