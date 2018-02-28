from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os
from lib.datasets.dataset_wrapper import DatasetWrapper
from utils.enums import Mode

class RandomDatasetWrapper(DatasetWrapper):
    """Wrapper which hold both the trainset and the validation set for cifar-10"""

    def __init__(self, *args, **kwargs):
        super(RandomDatasetWrapper, self).__init__(*args, **kwargs)

        self.train_random_dataset          = None
        self.train_random_eval_dataset     = None

        self.train_random_iterator         = None
        self.train_random_eval_iterator    = None

        self.train_random_handle           = None
        self.train_random_eval_handle      = None

    def set_datasets(self, X_train, y_train, X_test, y_test):
        """
        Setting different datasets based on the train/test data
        :param X_train: train data
        :param y_train: train labels
        :param X_test: test data
        :param y_test: test labels
        :return: None
        """
        # train set
        train_indices           = self.get_all_train_indices()
        train_images            = X_train[train_indices]
        train_labels            = y_train[train_indices]
        self.train_dataset      = self.set_transform('train'     , Mode.TRAIN, train_indices, train_images, train_labels)
        self.train_eval_dataset = self.set_transform('train_eval', Mode.EVAL , train_indices, train_images, train_labels)

        train_random_labels            = self.rand_gen.randint(self.num_classes, size=self.train_set_size, dtype=np.int32)
        self.train_random_dataset      = self.set_transform('train_random'     , Mode.TRAIN, train_indices, train_images, train_random_labels)
        self.train_random_eval_dataset = self.set_transform('train_random_eval', Mode.EVAL , train_indices, train_images, train_random_labels)

        save_path = os.path.join(self.prm.train.train_control.ROOT_DIR, 'train_random_labels.npy')
        self.log.info('saving train_random_labels to numpy file {}'.format(save_path))
        np.save(save_path, train_random_labels)

        # validation set
        validation_indices      = self.get_all_validation_indices()
        validation_images       = X_train[validation_indices]
        validation_labels       = y_train[validation_indices]
        self.validation_dataset = self.set_transform('validation', Mode.EVAL, validation_indices, validation_images, validation_labels)

        # test set
        test_indices            = range(X_test.shape[0])
        test_images             = X_test
        test_labels             = y_test
        self.test_dataset       = self.set_transform('test', Mode.EVAL, test_indices, test_images, test_labels)

    def build_iterators(self):
        super(RandomDatasetWrapper, self).build_iterators()
        self.train_random_iterator      = self.train_random_dataset.make_one_shot_iterator()
        self.train_random_eval_iterator = self.train_random_eval_dataset.make_initializable_iterator()

    def set_handles(self, sess):
        super(RandomDatasetWrapper, self).set_handles(sess)
        self.train_random_handle      = sess.run(self.train_random_iterator.string_handle())
        self.train_random_eval_handle = sess.run(self.train_random_eval_iterator.string_handle())

    def get_handle(self, name):
        if name == 'train_random':
            return self.train_random_handle
        elif name == 'train_random_eval':
            return self.train_random_eval_handle
        return super(RandomDatasetWrapper, self).get_handle(name)


