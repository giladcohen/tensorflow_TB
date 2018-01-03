from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from lib.base.agent_base import AgentBase
import lib.logger.logger as logger
import os
from utils import misc
import csv
import re
import tensorflow as tf
from lib.datasets.hooks.iterator_initializer_hook import IteratorInitializerHook

class DatasetWrapper(AgentBase):
    """Wrapper which hold both the trainset and the validation set for cifar-10"""
    #FIXME(gilad): consider using parent class for other datasets

    def __init__(self, name, prm, model):
        super(DatasetWrapper, self).__init__(name)
        self.prm = prm
        self.model = model
        self.dataset_name           = self.prm.dataset.DATASET_NAME
        # self.dataset_dir            = self.prm.dataset.DATASET_DIR
        self.train_set_size         = self.prm.dataset.TRAIN_SET_SIZE
        self.validation_set_size    = self.prm.dataset.VALIDATION_SET_SIZE
        self.test_set_size          = self.prm.dataset.TEST_SET_SIZE
        # self.train_images_dir       = self.prm.dataset.TRAIN_IMAGES_DIR
        # self.train_labels_file      = self.prm.dataset.TRAIN_LABELS_FILE
        # self.test_images_dir        = self.prm.dataset.TEST_IMAGES_DIR
        # self.test_labels_file       = self.prm.dataset.TEST_LABELS_FILE

        # self.train_dataset         = None
        self.train_iterator        = None
        self.train_input_hook      = None

        # self.validation_dataset    = None
        self.validation_iterator   = None
        self.validation_input_hook = None

        # self.test_dataset          = None
        self.test_iterator         = None
        self.test_input_hook       = None

        self.log = logger.get_logger(name)
        self.rand_gen = np.random.RandomState(prm.SUPERSEED)

        if self.validation_set_size is None:
            self.log.warning('Validation set size is None. Setting its size to 0')
            self.validation_set_size = 0
        self.train_validation_size  = self.train_set_size + self.validation_set_size

        self.train_validation_dict = {}
        self.initialize_datasets()

    def map_train_validation(self):
        """
        Sets the dictionary that maps each sample in the train set to 'train' or 'validation'
        :return: None
        """
        validation_indices = self.rand_gen.choice(range(self.train_validation_size), self.validation_set_size, replace=False)
        validation_indices = validation_indices.tolist()
        validation_indices.sort()

        for ind in range(self.train_validation_size):
            if ind in validation_indices:
                self.train_validation_dict[ind] = 'validation'
            else:
                self.train_validation_dict[ind] = 'train'
        dict_save_path = os.path.join(self.prm.train.train_control.ROOT_DIR, 'train_validation_dict.csv')
        with open(dict_save_path, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.train_validation_dict.items():
                writer.writerow([key, value])

    def initialize_datasets(self):
        """
        Sets the train/validation/test sets
        :return: None
        """
        self.map_train_validation()

        if self.dataset_name == 'cifar10':
            data = tf.keras.datasets.cifar10
        elif self.dataset_name == 'cifar100':
            data = tf.keras.datasets.cifar100
        else:
            err_str = 'dataset {} is not legal'.format(self.dataset_name)
            self.log.error(err_str)
            raise AssertionError(err_str)

        (X_train, y_train), (X_test, y_test) = data.load_data()

        if self.train_validation_size != X_train.shape[0]:
            err_str = 'train_set_size + validation_set_size = {} instead of {}'.format(self.train_validation_size, X_train.shape[0])
            self.log.error(err_str)
            raise AssertionError(err_str)

        (train_images, train_labels), (validation_images, validation_labels) = self.split_train_validation(X_train, y_train)
        (test_images, test_labels) = (X_test, y_test)

        # set datasets iterators
        self.train_iterator     , self.train_input_hook      = self.set_transform('train')
        self.validation_iterator, self.validation_input_hook = self.set_transform('validation')
        self.test_iterator      , self.test_input_hook       = self.set_transform('test')

    def split_train_validation(self):
        pass


    def set_transform(self, dataset_type):
        """
        Adding some transformation on a dataset
        :param dataset_type: 'train'/'validation'/'test'
        :return: None.
        """
        with tf.name_scope(dataset_type + '_data'):
            # feed all datasets with the same model placeholders:
            dataset = tf.data.Dataset.from_tensor_slices((self.model.images, self.model.labels))

            if dataset_type == 'train':
                dataset = dataset.shuffle(
                    buffer_size=self.train_set_size,
                    seed=self.prm.SUPERSEED,
                    reshuffle_each_iteration=True)
                dataset = dataset.batch(self.prm.train.train_control.TRAIN_BATCH_SIZE)
                dataset = dataset.repeat()

            iterator = dataset.make_initializable_iterator()

            # set runhok to initialize iterator
            iterator_initializer_hook = IteratorInitializerHook()
            # iterator_initializer_hook.iterator_initializer_func =
            #     lambda sess: sess.run(
            #         iterator.initializer,
            #         feed_dict={self.model.images: images,
            #                    labels_placeholder: labels})
            #
            return iterator, iterator_initializer_hook
