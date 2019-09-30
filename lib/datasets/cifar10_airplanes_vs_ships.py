
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from tensorflow_TB.lib.datasets.dataset_wrapper import DatasetWrapper
import tensorflow as tf


class CIFAR10_AirplanesVShips(DatasetWrapper):
    """Wrapper which hold both the trainset and the validation set for cifar-10"""

    def get_raw_data(self, dataset_name):
        """get data only for airplanes (0) and ships (8)
        :param dataset_name: the name of the dataset
        :return (X_train, y_train), (X_test, y_test) where
        X_train/test.shape = [?, H, W, 3], dtype=float32
        y_train/test.shape = [?]         , dtype=int
        """
        assert dataset_name == 'cifar10_airplanes_v_ships'
        assert self.num_classes == 2
        assert self.randomize_subset is True  # not supporting uneven dataset

        data = tf.keras.datasets.cifar10
        (X_train, y_train), (X_test, y_test) = data.load_data()
        num_samples_per_class = int(self.train_set_size / self.num_classes)

        train_indices = []
        test_indices  = []
        for cls in [0, 8]:
            possible_train_indices = np.where(y_train == cls)[0]
            new_train_indices = self.rand_gen.choice(possible_train_indices, num_samples_per_class, replace=False)
            train_indices += new_train_indices.tolist()
            test_indices  += np.where(y_test == cls)[0].tolist()

        train_indices.sort()
        test_indices.sort()
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        X_test  = X_test[test_indices]
        y_test  = y_test[test_indices]

        # replace label 0->0 and 8->1
        for i, label in enumerate(y_train):
            if label == 0:
                y_train[i] = 0
            elif label == 8:
                y_train[i] = 1
            else:
                err_str = 'y_train[{}] equals {} instead of 0 or 8'.format(i, label)
                self.log.error(err_str)
                raise AssertionError(err_str)

        for i, label in enumerate(y_test):
            if label == 0:
                y_test[i] = 0
            elif label == 8:
                y_test[i] = 1
            else:
                err_str = 'y_test[{}] equals {} instead of 0 or 8'.format(i, label)
                self.log.error(err_str)
                raise AssertionError(err_str)

        y_train = np.squeeze(y_train, axis=1)
        y_test  = np.squeeze(y_test, axis=1)

        if self.train_validation_size != X_train.shape[0]:
            err_str = 'train_set_size + validation_set_size = {} instead of {}'.format(self.train_validation_size, X_train.shape[0])
            self.log.error(err_str)
            raise AssertionError(err_str)

        if X_train.shape[-1] != self.num_channels or X_test.shape[-1] != self.num_channels:
            err_str = 'X_train.shape[-1] = {}. X_test.shape[-1] = {}. NUM_CHANNELS = {}' \
                .format(X_train.shape[-1], X_test.shape[-1], self.num_channels)
            self.log.error(err_str)
            raise AssertionError(err_str)

        return (X_train, y_train), (X_test, y_test)

