from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import darkon.darkon as darkon
import darkon_examples.cifar10_resnet.cifar10_input as cifar10_input
from tensorflow_TB.utils.misc import one_hot
from sklearn.model_selection import train_test_split
from copy import copy, deepcopy

class MyFeederValTest(darkon.InfluenceFeeder):
    def __init__(self, rand_gen, as_one_hot, val_inds=None, test_val_set=False):
        # load train data
        data, label = cifar10_input.prepare_train_data(padding_size=0)
        data /= 255.

        if val_inds is None:
            # here we split the data set to train and validation
            print('Feeder {} did not get val indices, therefore splitting trainset'.format(str(self)))
            indices = np.arange(data.shape[0])
            train_inds, val_inds = \
                train_test_split(indices, test_size=1000, random_state=rand_gen, shuffle=True, stratify=label)
        else:
            # val_inds were provided, so we need to infer all other indices
            train_inds = []
            # here we split the data set to train, validation, and test
            for ind in range(data.shape[0]):
                if ind not in val_inds:
                    train_inds.append(ind)
            train_inds = np.asarray(train_inds, dtype=np.int32)

        train_inds.sort()
        val_inds.sort()
        # save entire train data just for corner usage
        self.complete_data = data
        if as_one_hot:
            self.complete_label = one_hot(label.astype(np.int32), 10).astype(np.float32)
        else:
            self.complete_label = label

        # train data
        self.train_inds        = train_inds
        self.train_origin_data = data[train_inds]
        self.train_data        = data[train_inds]
        if as_one_hot:
            self.train_label = one_hot(label[train_inds].astype(np.int32), 10).astype(np.float32)
        else:
            self.train_label = label[train_inds]

        # validation data
        self.val_inds          = val_inds
        self.val_origin_data   = data[val_inds]
        self.val_data          = data[val_inds]
        if as_one_hot:
            self.val_label = one_hot(label[val_inds].astype(np.int32), 10).astype(np.float32)
        else:
            self.val_label = label[val_inds]

        if test_val_set:
            self.test_origin_data = self.val_origin_data
            self.test_data        = self.val_data
            self.test_label       = self.val_label
        else:
            data, label = cifar10_input.read_validation_data_wo_whitening()
            data /= 255.

            self.test_origin_data = data
            self.test_data        = data
            if as_one_hot:
                self.test_label = one_hot(label.astype(np.int32), 10).astype(np.float32)
            else:
                self.test_label = label

        self.train_batch_offset = 0

    def indices(self, indices):
        return self.complete_data[indices], self.complete_label[indices]

    def train_indices(self, indices):
        return self.train_data[indices], self.train_label[indices]

    def val_indices(self, indices):
        return self.val_data[indices], self.val_label[indices]

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

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
