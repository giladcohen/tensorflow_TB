from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import darkon
import darkon_examples.cifar10_resnet.cifar10_input as cifar10_input
from tensorflow_TB.utils.misc import one_hot
from sklearn.model_selection import train_test_split

class MyFeederVal(darkon.InfluenceFeeder):
    def __init__(self, as_one_hot, rand_gen):
        # load train data
        data, label = cifar10_input.prepare_train_data(padding_size=0)
        data /= 255.

        # here we split the data set to train and validation
        indices = np.arange(data.shape[0])
        data_train, data_val, label_train, label_val, indices_train, indices_val = \
            train_test_split(data, label, indices, test_size=1000, random_state=rand_gen, shuffle=True, stratify=label)

        # train data
        self.train_inds        = indices_train
        self.train_origin_data = data_train
        self.train_data        = data_train
        if as_one_hot:
            self.train_label = one_hot(label_train.astype(np.int32), 10).astype(np.float32)
        else:
            self.train_label = label_train

        # test data
        self.test_inds        = indices_val
        self.test_origin_data = data_val
        self.test_data        = data_val
        if as_one_hot:
            self.test_label = one_hot(label_val.astype(np.int32), 10).astype(np.float32)
        else:
            self.test_label = label_val

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
