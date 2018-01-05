from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os
from lib.base.agent_base import AgentBase
import lib.logger.logger as logger
import csv
import tensorflow as tf

class DatasetWrapper(AgentBase):
    """Wrapper which hold both the trainset and the validation set for cifar-10"""
    #FIXME(gilad): consider using parent class for other datasets

    def __init__(self, name, prm):
        super(DatasetWrapper, self).__init__(name)
        self.prm = prm
        self.dataset_name          = self.prm.dataset.DATASET_NAME
        self.train_set_size        = self.prm.dataset.TRAIN_SET_SIZE
        self.validation_set_size   = self.prm.dataset.VALIDATION_SET_SIZE
        self.test_set_size         = self.prm.dataset.TEST_SET_SIZE
        self.H                     = self.prm.network.IMAGE_HEIGHT
        self.W                     = self.prm.network.IMAGE_WIDTH
        self.train_batch_size      = self.prm.train.train_control.TRAIN_BATCH_SIZE
        self.eval_batch_size       = self.prm.train.train_control.EVAL_BATCH_SIZE
        self.log                   = logger.get_logger(name)
        self.rand_gen              = np.random.RandomState(prm.SUPERSEED)

        self.train_validation_dict = {}

        self.train_dataset         = None
        self.validation_dataset    = None
        self.test_dataset          = None

        self.iterator              = None
        self.train_iterator        = None  # static iterator for train only
        self.validation_iterator   = None  # dynamic iterator for validation. need to reinitialize
        self.test_iterator         = None  # dynamic iterator for test. need to reinitialize

        self.handle                = None
        self.train_handle          = None
        self.validation_handle     = None
        self.test_handle           = None

        self.next_minibatch        = None  # this is the output of iterator.get_next()

        if self.validation_set_size is None:
            self.log.warning('Validation set size is None. Setting its size to 0')
            self.validation_set_size = 0
        self.train_validation_size  = self.train_set_size + self.validation_set_size

    def build(self):
        """
        Build the datasets and iterators. Session must be provided for the handles.
        :param sess: session
        :return: None
        """
        self.map_train_validation()
        self.build_datasets()
        self.build_iterators()

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

    def build_datasets(self):
        """
        Sets the train/validation/test datasets
        :return: None
        """
        if self.dataset_name == 'cifar10':
            data = tf.keras.datasets.cifar10
        elif self.dataset_name == 'cifar100':
            data = tf.keras.datasets.cifar100
        else:
            err_str = 'dataset {} is not legal'.format(self.dataset_name)
            self.log.error(err_str)
            raise AssertionError(err_str)

        (X_train, y_train), (X_test, y_test) = data.load_data()
        y_train = np.squeeze(y_train, axis=1)
        y_test  = np.squeeze(y_test , axis=1)

        if self.train_validation_size != X_train.shape[0]:
            err_str = 'train_set_size + validation_set_size = {} instead of {}'.format(self.train_validation_size, X_train.shape[0])
            self.log.error(err_str)
            raise AssertionError(err_str)

        (train_images, train_labels), (validation_images, validation_labels) = self.split_train_validation(X_train, y_train)
        (test_images, test_labels) = (X_test, y_test)

        # generate datasets
        self.train_dataset       = self.set_transform('train'     , train_images     , train_labels)
        self.validation_dataset  = self.set_transform('validation', validation_images, validation_labels)
        self.test_dataset        = self.set_transform('test'      , test_images      , test_labels)

    def build_iterators(self):
        """
        Sets the train/validation/test iterators
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
        self.validation_handle = sess.run(self.validation_iterator.string_handle())
        self.test_handle       = sess.run(self.test_iterator.string_handle())

    def split_train_validation(self, X_train, y_train):
        """
        Splitting the train set to train and validation sets
        :param X_train: images
        :param y_train: labels
        :return: (train_images, train_labels), (validation_images, validation_labels)
        """

        train_indices      = [k for k,v in self.train_validation_dict.iteritems() if v == 'train']
        validation_indices = [k for k,v in self.train_validation_dict.iteritems() if v == 'validation']
        train_images      = X_train[train_indices]
        train_labels      = y_train[train_indices]
        validation_images = X_train[validation_indices]
        validation_labels = y_train[validation_indices]

        if train_images.shape[0] != self.train_set_size or validation_images.shape[0] != self.validation_set_size:
            err_str = 'The train/val set size is ({},{}) instead of ({},{})'.format(
                train_images.shape[0], validation_images.shape[0], self.train_set_size, self.validation_set_size)
            self.log.error(err_str)
            raise AssertionError(err_str)

        return (train_images, train_labels), (validation_images, validation_labels)

    def set_transform(self, dataset_type, images, labels):
        """
        Adding some transformation on a dataset
        :param dataset_type: 'train'/'validation'/'test'
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

        if dataset_type == 'train':
            batch_size = self.prm.train.train_control.TRAIN_BATCH_SIZE
        else:
            batch_size = self.prm.train.train_control.EVAL_BATCH_SIZE

        with tf.name_scope(dataset_type + '_data'):
            # feed all datasets with the same model placeholders:
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            dataset = dataset.map(map_func=_cast, num_parallel_calls=batch_size)

            if dataset_type == 'train':
                dataset = dataset.map(map_func=_augment, num_parallel_calls=batch_size)
                dataset = dataset.shuffle(
                    buffer_size=batch_size,
                    seed=self.prm.SUPERSEED,
                    reshuffle_each_iteration=True)
                dataset = dataset.prefetch(5 * batch_size)
                dataset = dataset.repeat()

            dataset = dataset.batch(batch_size)

            return dataset

    def get_mini_batch(self, dataset_type, sess):
        """
        Get a session and returns the next training batch
        :param sess: Session
        :return: next training batch
        """
        if dataset_type == 'train':
            handle = self.train_handle
        elif dataset_type == 'validation':
            handle = self.validation_handle
        elif dataset_type == 'test':
            handle = self.test_handle
        else:
            err_str = 'calling get_mini_batch with illegal dataset type ({})'.format(dataset_type)
            self.log.error(err_str)
            raise AssertionError(err_str)

        images, labels = sess.run(self.next_minibatch, feed_dict={self.handle: handle})
        return images, labels

    def print_stats(self):
        """print dataset parameters"""
        self.log.info('Dataset parameters:')
        self.log.info(' DATASET_NAME: {}'.format(self.dataset_name))
        self.log.info(' TRAIN_SET_SIZE: {}'.format(self.train_set_size))
        self.log.info(' VALIDATION_SET_SIZE: {}'.format(self.validation_set_size))
        self.log.info(' TEST_SET_SIZE: {}'.format(self.test_set_size))


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