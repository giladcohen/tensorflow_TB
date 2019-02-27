from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import darkon
import tensorflow as tf
import numpy as np
from lib.base.agent_base import AgentBase
from utils.misc import err_n_assert

class Feeder(darkon.InfluenceFeeder, AgentBase):
    def __init__(self, name, prm):
        # load train data
        self.name = name
        self.prm = prm
        super(Feeder, self).__init__(name)

        self.dataset_name = self.prm.dataset.DATASET_NAME

    def build(self):
        """
        Builds the train/eval/test data and labels
        """
        self.build_dataset()
        self.reset()

    def build_dataset(self):
        # reading the dataset from the parameters
        if 'cifar100' in self.dataset_name:
            data = tf.keras.datasets.cifar100
        elif 'cifar10' in self.dataset_name:
            data = tf.keras.datasets.cifar10
        elif 'mnist' in self.dataset_name:
            data = tf.keras.datasets.mnist
        else:
            data = None
            err_n_assert(self.log, 'dataset {} is not legal'.format(self.dataset_name))

        (X_train, y_train), (X_test, y_test) = data.load_data()

        if 'cifar' in self.dataset_name:
            y_train = np.squeeze(y_train, axis=1)
            y_test  = np.squeeze(y_test , axis=1)
        if 'mnist' in self.dataset_name:
            X_train = np.expand_dims(X_train, axis=-1)
            X_test  = np.expand_dims(X_test, axis=-1)

        self.train_origin_data = X_train
        self.train_data        = self.whitening_image(X_train)
        self.train_label       = y_train

        self.test_origin_data  = X_test
        self.test_data         = self.whitening_image(X_test)
        self.test_label        = y_test

    def test_indices(self, indices):
        return self.test_data[indices], self.test_label[indices]

    def train_batch(self, batch_size):
        # calculate offset
        start = self.train_batch_offset
        end = start + batch_size
        self.train_batch_offset += batch_size

        return self.train_data[start:end, ...], self.train_label[start:end, ...]

    def train_one(self, idx):
        return self.train_data[idx, ...], self.train_label[idx, ...]

    def reset(self):
        self.train_batch_offset = 0

    def whitening_image(self, image_np):
        '''
        Performs per_image_whitening
        :param image_np: a 4D numpy array representing a batch of images
        :return: the image numpy array after whitened
        '''
        im_shape = image_np.shape
        for i in range(im_shape[0]):
            mean = np.mean(image_np[i, ...])
            # Use adjusted standard deviation here, in case the std == 0.
            std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(im_shape[1] * im_shape[2] * im_shape[3])])
            image_np[i,...] = (image_np[i, ...] - mean) / std
        return image_np
