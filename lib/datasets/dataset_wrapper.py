from __future__ import absolute_import

import lib.logger.logger as logger
import os
from utils import misc

class DatasetWrapper(object):
    """Wrapper which hold both the trainset and the validation set for cifar-10"""
    #FIXME(gilad): consider using parent class for other datasets

    def __init__(self, name, prm, train_dataset, validation_dataset):
        self.name = name
        self.prm = prm
        self.train_dataset      = train_dataset
        self.validation_dataset = validation_dataset
        self.log = logger.get_logger(name)

        self.dataset_name           = self.prm.dataset.DATASET_NAME
        self.dataset_dir            = self.prm.dataset.DATASET_DIR
        self.train_set_size         = self.prm.dataset.TRAIN_SET_SIZE
        self.validation_set_size    = self.prm.dataset.VALIDATION_SET_SIZE
        self.train_images_dir       = self.prm.dataset.TRAIN_IMAGES_DIR
        self.train_labels_file      = self.prm.dataset.TRAIN_LABELS_FILE
        self.validation_images_dir  = self.prm.dataset.VALIDATION_IMAGES_DIR
        self.validation_labels_file = self.prm.dataset.VALIDATION_LABELS_FILE

        self.verify_dataset()

    def __str__(self):
        return self.name

    def print_stats(self):
        """print dataset parameters"""
        self.log.info('Dataset parameters:')
        self.log.info(' DATASET_NAME: {}'.format(self.dataset_name))
        self.log.info(' DATASET_DIR: {}'.format(self.dataset_dir))
        self.log.info(' TRAIN_SET_SIZE: {}'.format(self.train_set_size))
        self.log.info(' VALIDATION_SET_SIZE: {}'.format(self.validation_set_size))
        self.log.info(' TRAIN_IMAGES_DIR: {}'.format(self.train_images_dir))
        self.log.info(' TRAIN_LABELS_FILE: {}'.format(self.train_labels_file))
        self.log.info(' VALIDATION_IMAGES_DIR: {}'.format(self.validation_images_dir))
        self.log.info(' VALIDATION_LABELS_FILE: {}'.format(self.validation_labels_file))

        self.train_dataset.print_stats()
        self.validation_dataset.print_stats()

    def get_mini_batch_train(self, *args, **kwargs):
        return self.train_dataset.get_mini_batch(*args, **kwargs)

    def get_mini_batch_validate(self, *args, **kwargs):
        return self.validation_dataset.get_mini_batch(*args, **kwargs)

    # def verify_dataset(self):
    #     if self.train_images_dir       is None or \
    #        self.train_labels_file      is None or \
    #        self.validation_images_dir  is None or \
    #        self.validation_labels_file is None or \
    #        self.dataset_dir            is None:
    #         err_str = 'One or more of the train/validation paths is None'
    #         self.log.error(err_str)
    #         raise AssertionError(err_str)
    #
    #     dirname = os.path.dirname
    #     if dirname(self.train_images_dir)       != self.dataset_dir or \
    #        dirname(self.train_labels_file)      != self.dataset_dir or \
    #        dirname(self.validation_images_dir)  != self.dataset_dir or \
    #        dirname(self.validation_labels_file) != self.dataset_dir:
    #         err_str = 'One or more of the train/validation paths is not in {}'.format(self.dataset_dir)
    #         self.log.error(err_str)
    #         raise AssertionError(err_str)
    #
    #     if not os.path.exists(self.dataset_dir):
    #         os.makedirs(self.train_images_dir)
    #         os.makedirs(self.validation_images_dir)
    #         self.log.info('Creating {} dataset into {}. This may take a while'.format(self.dataset_name, self.dataset_dir))
    #         misc.save_cifar10_to_disk(self.train_images_dir, self.train_labels_file,
    #                                   self.validation_images_dir, self.validation_labels_file)
    #         self.log.info('dataset {} was successfully written to {}.'.format(self.dataset_name, self.dataset_dir))
