from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import darkon
import darkon_examples.cifar10_resnet.cifar10_input as cifar10_input
from tensorflow_TB.utils.misc import one_hot

class MyFeeder(darkon.InfluenceFeeder):
    def __init__(self, as_one_hot):
        # load train data
        data, label = cifar10_input.prepare_train_data(padding_size=0)
        self.train_origin_data = data / 255.
        self.train_data        = data / 255.
        if as_one_hot:
            label = label.astype(np.int32)
            self.train_label = one_hot(label, 10).astype(np.float32)
        else:
            self.train_label = label

        # load test data
        data, label = cifar10_input.read_validation_data_wo_whitening()
        self.test_origin_data = data / 255.
        self.test_data        = data / 255.
        if as_one_hot:
            label = label.astype(np.int32)
            self.test_label = one_hot(label, 10).astype(np.float32)
        else:
            self.test_label = label

        self.train_batch_offset = 0

    def train_indices(self, indices):
        return self.train_data[indices], self.train_label[indices]

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
