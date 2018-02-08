from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os
from lib.base.agent_base import AgentBase
import lib.logger.logger as logger
import csv
import tensorflow as tf
from utils.enums import Mode

class DatasetWrapper(AgentBase):
    """Wrapper which hold both the trainset and the validation set for cifar-10"""

    def __init__(self, name, prm):
        super(DatasetWrapper, self).__init__(name)
        self.prm = prm
        self.log = logger.get_logger(name)
        self.dataset_name             = self.prm.dataset.DATASET_NAME
        self.train_set_size           = self.prm.dataset.TRAIN_SET_SIZE
        self.validation_set_size      = self.prm.dataset.VALIDATION_SET_SIZE
        self.test_set_size            = self.prm.dataset.TEST_SET_SIZE
        self.train_validation_map_ref = self.prm.dataset.TRAIN_VALIDATION_MAP_REF
        self.H                        = self.prm.network.IMAGE_HEIGHT
        self.W                        = self.prm.network.IMAGE_WIDTH
        self.train_batch_size         = self.prm.train.train_control.TRAIN_BATCH_SIZE
        self.eval_batch_size          = self.prm.train.train_control.EVAL_BATCH_SIZE
        self.rand_gen                 = np.random.RandomState(prm.SUPERSEED)

        self.train_validation_info    = []

        self.train_dataset            = None
        self.train_eval_dataset       = None
        self.validation_dataset       = None
        self.test_dataset             = None

        self.iterator                 = None
        self.train_iterator           = None  # static iterator for train only
        self.train_eval_iterator      = None  # dynamic iterator for train evaluation. need to reinitialize
        self.validation_iterator      = None  # dynamic iterator for validation. need to reinitialize
        self.test_iterator            = None  # dynamic iterator for test. need to reinitialize

        self.handle                   = None
        self.train_handle             = None
        self.train_eval_handle        = None
        self.validation_handle        = None
        self.test_handle              = None

        self.next_minibatch           = None  # this is the output of iterator.get_next()

        if self.validation_set_size is None:
            self.log.warning('Validation set size is None. Setting its size to 0')
            self.validation_set_size = 0
        self.train_validation_size  = self.train_set_size + self.validation_set_size

    def build(self):
        """
        Build the datasets and iterators.
        :return: None
        """
        self.map_train_validation()
        self.build_datasets()
        self.build_iterators()

    def map_train_validation(self):
        """
        Sets the dictionary that maps each sample in the train set to 'train' or 'validation'
        :return: None. Updates self.train_validation_info.
        """
        # optionally load train-validation mapping reference reference
        if self.train_validation_map_ref is not None:
            self.load_data_info()
        else:
            self.set_data_info()
        self.save_data_info()

    def load_data_info(self):
        """Loading self.train_validation_info from ref"""
        self.log.info('train_validation_map_ref was given. loading csv {}'.format(self.train_validation_map_ref))
        with open(self.train_validation_map_ref) as csv_file:
            self.train_validation_info = \
                [{'index': int(row['index']), 'dataset': row['dataset'], 'in_pool': row['in_pool'] == 'True'}
                 for row in csv.DictReader(csv_file, skipinitialspace=True)]

    def set_data_info(self):
        """Updating self.train_validation_info dictionary is one is not loaded from ref"""
        self.log.info('train_validation_map_ref is None. Creating new mapping')

        validation_indices = self.rand_gen.choice(range(self.train_validation_size), self.validation_set_size, replace=False)
        validation_indices = validation_indices.tolist()
        validation_indices.sort()

        for ind in range(self.train_validation_size):
            if ind in validation_indices:
                self.train_validation_info.append({'index': ind,
                                                   'dataset': 'validation',
                                                   'in_pool': False})
            else:
                self.train_validation_info.append({'index': ind,
                                                   'dataset': 'train',
                                                   'in_pool': False})  # 'in_pool' is only used for active learning.

    def save_data_info(self):
        """Saving self.train_validation_info into disk"""
        info_save_path = os.path.join(self.prm.train.train_control.ROOT_DIR, 'train_validation_info.csv')
        self.log.info('saving train-validation mapping csv file to {}'.format(info_save_path))
        keys = self.train_validation_info[0].keys()
        with open(info_save_path, 'wb') as csv_file:
            dict_writer = csv.DictWriter(csv_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.train_validation_info)

    def get_raw_data(self, dataset_name):
        """This function get the string dataset_name (such as cifar10 or cifar100) and returns images and labels
        :param dataset_name: the name of the dataset
        :return (X_train, y_train), (X_test, y_test) where
        X_train/test.shape = [?, H, W, 3], dtype=float32
        y_train/test.shape = [?]         , dtype=int
        """

        if 'cifar100' in dataset_name:
            data = tf.keras.datasets.cifar100
        elif 'cifar10' in dataset_name:
            data = tf.keras.datasets.cifar10
        else:
            err_str = 'dataset {} is not legal'.format(dataset_name)
            self.log.error(err_str)
            raise AssertionError(err_str)

        (X_train, y_train), (X_test, y_test) = data.load_data()
        y_train = np.squeeze(y_train, axis=1)
        y_test  = np.squeeze(y_test , axis=1)

        if self.train_validation_size != X_train.shape[0]:
            err_str = 'train_set_size + validation_set_size = {} instead of {}'.format(self.train_validation_size, X_train.shape[0])
            self.log.error(err_str)
            raise AssertionError(err_str)

        return (X_train, y_train), (X_test, y_test)

    def build_datasets(self):
        """
        Sets the train/validation/test datasets
        :return: None
        """
        (X_train, y_train), (X_test, y_test) = self.get_raw_data(self.dataset_name)
        self.set_datasets(X_train, y_train, X_test, y_test)

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
        train_indices           = [element['index'] for element in self.train_validation_info if element['dataset'] == 'train']
        train_images            = X_train[train_indices]
        train_labels            = y_train[train_indices]
        self.train_dataset      = self.set_transform('train', Mode.TRAIN, train_images, train_labels)

        # train eval set, for evaluation only
        self.train_eval_dataset = self.set_transform('train_eval', Mode.EVAL, train_images, train_labels)

        # validation set
        validation_indices      = [element['index'] for element in self.train_validation_info if element['dataset'] == 'validation']
        validation_images       = X_train[validation_indices]
        validation_labels       = y_train[validation_indices]
        self.validation_dataset = self.set_transform('validation', Mode.EVAL, validation_images, validation_labels)

        # test set
        test_images             = X_test
        test_labels             = y_test
        self.test_dataset       = self.set_transform('test', Mode.EVAL, test_images, test_labels)

    def build_iterators(self):
        """
        Sets the train/validation/test/train_eval iterators
        :return: None
        """

        # A feedable iterator is defined by a handle placeholder and its structure. We
        # could use the `output_types` and `output_shapes` properties of either
        # `training_dataset` or `validation_dataset` here, because they have
        # identical structure.
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)
        self.next_minibatch = self.iterator.get_next()

        # generate iterators
        self.train_iterator      = self.train_dataset.make_one_shot_iterator()
        self.train_eval_iterator = self.train_eval_dataset.make_initializable_iterator()
        self.validation_iterator = self.validation_dataset.make_initializable_iterator()
        self.test_iterator       = self.test_dataset.make_initializable_iterator()

    def set_handles(self, sess):
        """
        set the handles. Must be called from the trainer/tester, using a session
        :param sess: session
        :return: None
        """
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        self.train_handle      = sess.run(self.train_iterator.string_handle())
        self.train_eval_handle = sess.run(self.train_eval_iterator.string_handle())
        self.validation_handle = sess.run(self.validation_iterator.string_handle())
        self.test_handle       = sess.run(self.test_iterator.string_handle())

    def set_transform(self, name, mode, images, labels):
        """
        Adding some transformation on a dataset
        :param name: name of the dataset (string). Examples: 'train'/'validation'/'test/train_eval'
        :param mode: Mode (TRAIN/EVAL/PREDICT)
        :return: None.
        """

        def _augment(image, label):
            """
            :param image: input image
            :param label: input label
            :return: An augmented image with label
            """
            image = tf.image.resize_image_with_crop_or_pad(
                image, self.H + 4, self.W + 4)  # padding with 2 zeros at every side
            image = tf.random_crop(image, [self.H, self.H, 3], seed=self.prm.SUPERSEED)
            image = tf.image.random_flip_left_right(image, seed=self.prm.SUPERSEED)
            # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
            # image = tf.image.random_brightness(image, max_delta=63. / 255.)
            # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            return image, label

        def _cast(image, label):
            """
            Casting the image to tf.float32 and the label to tf.int32
            :param image: input image
            :param label: input labels
            :return: cated image and casted label
            """
            image = tf.cast(image, tf.float32)
            label = tf.cast(label, tf.int32)
            return image, label

        if mode == Mode.TRAIN:
            batch_size = self.prm.train.train_control.TRAIN_BATCH_SIZE
        else:
            batch_size = self.prm.train.train_control.EVAL_BATCH_SIZE

        with tf.name_scope(name + '_data'):
            # feed all datasets with the same model placeholders:
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            dataset = dataset.map(map_func=_cast, num_parallel_calls=batch_size)

            if mode == Mode.TRAIN:
                dataset = dataset.map(map_func=_augment, num_parallel_calls=batch_size)
                dataset = dataset.shuffle(
                    buffer_size=batch_size,
                    seed=self.prm.SUPERSEED,
                    reshuffle_each_iteration=True)
                dataset = dataset.prefetch(5 * batch_size)
                dataset = dataset.repeat()

            dataset = dataset.batch(batch_size)

            return dataset

    def get_mini_batch(self, name, sess):
        """
        Get a session and returns the next training batch
        :param name: the name of the dataset
        :param sess: Session
        :return: next training batch
        """
        handle = self.get_handle(name)
        images, labels = sess.run(self.next_minibatch, feed_dict={self.handle: handle})
        return images, labels

    def get_handle(self, name):
        """Getting an iterator handle based on dataset name
        :param name: name of the dataset (string). e.g., 'train', 'train_eval', 'validation', 'test', etc.
        """
        if name == 'train':
            return self.train_handle
        elif name == 'train_eval':
            return self.train_eval_handle
        elif name == 'validation':
            return self.validation_handle
        elif name == 'test':
            return self.test_handle

        err_str = 'calling get_mini_batch with illegal dataset name ({})'.format(name)
        self.log.error(err_str)
        raise AssertionError(err_str)

    def print_stats(self):
        """print dataset parameters"""
        self.log.info('Dataset parameters:')
        self.log.info(' DATASET_NAME: {}'.format(self.dataset_name))
        self.log.info(' TRAIN_SET_SIZE: {}'.format(self.train_set_size))
        self.log.info(' VALIDATION_SET_SIZE: {}'.format(self.validation_set_size))
        self.log.info(' TEST_SET_SIZE: {}'.format(self.test_set_size))
        self.log.info(' TRAIN_VALIDATION_MAP_REF: {}'.format(self.train_validation_map_ref))



# debug
# import os
# import sys
# cwd = os.getcwd() # tensorflow-TB
# sys.path.insert(0, cwd)
# from utils.parameters import Parameters
#
# prm_file = 'examples/train_simple_debug.ini'
# prm = Parameters()
# prm.override(prm_file)
# name = 'dataset_wrapper'
#
# dataset = DatasetWrapper(name, prm)
# dataset.build()
