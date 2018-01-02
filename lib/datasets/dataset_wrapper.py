from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import lib.logger.logger as logger
import os
from utils import misc
import csv
import re

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

class DatasetWrapper(object):
    """Wrapper which hold both the trainset and the validation set for cifar-10"""
    #FIXME(gilad): consider using parent class for other datasets

    def __init__(self, name, prm):
        self.name = name
        self.prm = prm
        self.train_dataset      = None
        self.validation_dataset = None
        self.test_dataset       = None
        self.log = logger.get_logger(name)
        self.rand_gen = np.random.RandomState(prm.SUPERSEED)

        self.dataset_name           = self.prm.dataset.DATASET_NAME
        self.dataset_dir            = self.prm.dataset.DATASET_DIR
        self.train_set_size         = self.prm.dataset.TRAIN_SET_SIZE
        self.validation_set_size    = self.prm.dataset.VALIDATION_SET_SIZE
        self.test_set_size          = self.prm.dataset.TEST_SET_SIZE
        self.train_images_dir       = self.prm.dataset.TRAIN_IMAGES_DIR
        self.train_labels_file      = self.prm.dataset.TRAIN_LABELS_FILE
        self.test_images_dir        = self.prm.dataset.TEST_IMAGES_DIR
        self.test_labels_file       = self.prm.dataset.TEST_LABELS_FILE

        if self.validation_set_size is None:
            self.log.warning('Validation set size is None. Setting its size to 0')
            self.validation_set_size = 0

        self.verify_dataset()

        # get the dataset for future assign to train/validation/test sets
        self.train_validation_images_list , self.train_validation_labels = self.load_dataset('train')
        self.test_images_list             , self.test_labels             = self.load_dataset('test')

    def load_dataset(self, dataset_type):
        images_dir  = self.train_images_dir if dataset_type == 'train' else self.test_images_dir
        labels_file =  self.train_labels_file if dataset_type == 'train' else self.test_labels_file
        size = (self.train_set_size + self.validation_set_size) if dataset_type == 'train' else self.test_set_size

        images_list = []
        local_list = sorted(os.listdir(images_dir), key=numericalSort)
        for file in local_list:
            images_list.append(os.path.join(images_dir, file))

        labels = np.empty([size], dtype=np.int)
        tmp_list = open(labels_file).read().splitlines()
        for i, val in enumerate(tmp_list):
            labels[i] = int(val)

        return images_list, labels

    def __str__(self):
        return self.name

    def print_stats(self):
        """print dataset parameters"""
        self.log.info('Dataset parameters:')
        self.log.info(' DATASET_NAME: {}'.format(self.dataset_name))
        self.log.info(' DATASET_DIR: {}'.format(self.dataset_dir))
        self.log.info(' TRAIN_SET_SIZE: {}'.format(self.train_set_size))
        self.log.info(' VALIDATION_SET_SIZE: {}'.format(self.validation_set_size))
        self.log.info(' TEST_SET_SIZE: {}'.format(self.test_set_size))
        self.log.info(' TRAIN_IMAGES_DIR: {}'.format(self.train_images_dir))
        self.log.info(' TRAIN_LABELS_FILE: {}'.format(self.train_labels_file))
        self.log.info(' TEST_IMAGES_DIR: {}'.format(self.test_images_dir))
        self.log.info(' TEST_LABELS_FILE: {}'.format(self.test_labels_file))

        self.train_dataset.print_stats()
        self.validation_dataset.print_stats()

    def get_mini_batch_train(self, *args, **kwargs):
        return self.train_dataset.get_mini_batch(*args, **kwargs)

    def get_mini_batch_validate(self, *args, **kwargs):
        return self.validation_dataset.get_mini_batch(*args, **kwargs)

    def verify_dataset(self):
        if self.train_images_dir       is None or \
           self.train_labels_file      is None or \
           self.test_images_dir        is None or \
           self.test_labels_file       is None or \
           self.dataset_dir            is None:
            err_str = 'One or more of the train/test paths is None'
            self.log.error(err_str)
            raise AssertionError(err_str)

        dirname = os.path.dirname
        if dirname(self.train_images_dir)       != self.dataset_dir or \
           dirname(self.train_labels_file)      != self.dataset_dir or \
           dirname(self.test_images_dir)  != self.dataset_dir or \
           dirname(self.test_labels_file) != self.dataset_dir:
            err_str = 'One or more of the train/test paths is not in {}'.format(self.dataset_dir)
            self.log.error(err_str)
            raise AssertionError(err_str)

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.train_images_dir)
            os.makedirs(self.test_images_dir)
            self.log.info('Creating {} dataset into {}. This may take a while'.format(self.dataset_name, self.dataset_dir))
            misc.save_dataset_to_disk(self.dataset_name,
                                      self.train_images_dir, self.train_labels_file,
                                      self.test_images_dir, self.test_labels_file)
            self.log.info('dataset {} was successfully written to {}.'.format(self.dataset_name, self.dataset_dir))

    def set_train_dataset(self, train_dataset):
        self.train_dataset = train_dataset

    def set_validation_dataset(self, validation_dataset):
        self.validation_dataset = validation_dataset

    def set_test_dataset(self, test_dataset):
        self.test_dataset             = test_dataset
        self.test_dataset.images_list = self.test_images_list
        self.test_dataset.labels      = self.test_labels

    def split_train_validation(self):
        """
        Splitting the train set to train and validation
        :return: None
        """
        #TODO(gilad): Support active learning with validation set in the future

        # Get all the info from the train set
        pool        = range(self.train_set_size + self.validation_set_size)
        images_list = self.train_validation_images_list
        labels      = self.train_validation_labels

        # Randomize the validation set indices
        validation_indices = self.rand_gen.choice(pool, self.validation_set_size, replace=False)
        validation_indices = validation_indices.tolist()
        validation_indices.sort()

        train_validation_dict = {}
        for ind in pool:
            if ind in validation_indices:
                train_validation_dict[ind] = 'validation'
            else:
                train_validation_dict[ind] = 'train'

        # Constructing new train/validation pool/data/labels
        train_pool             = []
        validation_pool        = []
        train_images_list      = []
        validation_images_list = []
        train_labels           = np.empty([self.train_set_size]     , dtype=np.int)
        validation_labels      = np.empty([self.validation_set_size], dtype=np.int)

        # Assiging data to train/validation set
        train_cnt      = 0
        validation_cnt = 0
        for ind in sorted(train_validation_dict.keys()):
            if train_validation_dict[ind] == 'train':
                train_pool.append(ind)
                train_images_list.append(images_list[ind])
                train_labels[train_cnt] = labels[ind]
                train_cnt += 1
            elif train_validation_dict[ind] == 'validation':
                validation_pool.append(ind)
                validation_images_list.append(images_list[ind])
                validation_labels[validation_cnt] = labels[ind]
                validation_cnt += 1
            else:
                err_str = 'the value {} is not legal in dictionary train_validation_dict'.format(train_validation_dict[ind])
                self.log.error(err_str)
                raise AssertionError(err_str)

        if train_cnt != self.train_set_size or validation_cnt != self.validation_set_size:
            err_str = 'got train_cnt={}, validation_cnt={} instead of {}/{}' \
                .format(train_cnt, validation_cnt, self.train_set_size, self.validation_set_size)
            self.log.error(err_str)
            raise AssertionError(err_str)

        self.train_dataset.pool             = train_pool
        self.train_dataset.images_list      = train_images_list
        self.train_dataset.labels           = train_labels

        self.validation_dataset.pool        = validation_pool
        self.validation_dataset.images_list = validation_images_list
        self.validation_dataset.labels      = validation_labels

        self.log.info('Done splitting train/validation sets to {}/{} size'.format(len(train_pool), len(validation_pool)))

        dict_save_path = os.path.join(self.prm.train.train_control.ROOT_DIR, 'train_validation_dict.csv')
        with open(dict_save_path, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in train_validation_dict.items():
                writer.writerow([key, value])
