from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from lib.datasets.dataset_wrapper import DatasetWrapper
from utils.enums import Mode

class SemiSupervisedDatasetWrapper(DatasetWrapper):
    def __init__(self, *args, **kwargs):
        super(SemiSupervisedDatasetWrapper, self).__init__(*args, **kwargs)

        self.unsupervised_percentage       = self.prm.train.train_control.semi_supervised.UNSUPERVISED_PERCENTAGE
        self.unsupervised_percentage_batch = self.prm.train.train_control.semi_supervised.UNSUPERVISED_PERCENTAGE_BATCH

        self.unpool_set_size               = int(self.unsupervised_percentage * self.train_set_size / 100)
        self.pool_set_size                 = self.train_set_size - self.unpool_set_size

        self.unpool_batch_size             = int(self.unsupervised_percentage_batch * self.train_batch_size / 100)
        self.pool_batch_size               = self.train_batch_size - self.unpool_batch_size

        self.train_unpool_soft_labels      = np.zeros([self.unpool_set_size, self.num_classes], dtype=np.float32)
        self.soft_labels_ready             = False  # whether or not we are good to use the soft label predictions

        self.train_pool_dataset            = None
        self.train_pool_init_dataset       = None  # only for initialization
        self.train_pool_eval_dataset       = None
        self.train_unpool_dataset          = None
        self.train_unpool_eval_dataset     = None

        self.train_pool_iterator           = None
        self.train_pool_init_iterator      = None
        self.train_pool_eval_iterator      = None
        self.train_unpool_iterator         = None
        self.train_unpool_eval_iterator    = None

        self.train_pool_handle             = None
        self.train_pool_init_handle        = None
        self.train_pool_eval_handle        = None
        self.train_unpool_handle           = None
        self.train_unpool_eval_handle      = None

    def set_data_info(self):
        """
        There is no ref to load data info from. Therefore we need to construct a dataset with pool size
        """
        # We start by setting self.train_validation_info as parent
        super(SemiSupervisedDatasetWrapper, self).set_data_info()

        # from all the train samples, choose only pool_set_size samples
        train_indices = self.get_all_train_indices()
        train_pool_indices = self.rand_gen.choice(train_indices, self.pool_set_size, replace=False)
        train_pool_indices = train_pool_indices.tolist()
        train_pool_indices.sort()
        for sample in self.train_validation_info:
            if sample['index'] in train_pool_indices:
                sample['in_pool'] = True

    def get_all_unpool_train_indices(self):
        """
        :return: all unpooled train indices, with 'in_pool'=False
        """
        indices = []
        for sample in self.train_validation_info:
            if sample['dataset'] == 'train' and not sample['in_pool']:
                indices.append(sample['index'])
        indices.sort()
        return indices

    def get_all_pool_train_indices(self):
        """
        :return: all pooled train indices, with 'in_pool'=True
        """
        indices = []
        for sample in self.train_validation_info:
            if sample['dataset'] == 'train' and sample['in_pool']:
                indices.append(sample['index'])
        indices.sort()
        return indices

    def set_datasets(self, X_train, y_train, X_test, y_test):
        super(SemiSupervisedDatasetWrapper, self).set_datasets(X_train, y_train, X_test, y_test)

        # train_pool_set
        train_pool_indices             = self.get_all_pool_train_indices()
        train_pool_images              = X_train[train_pool_indices]
        train_pool_labels              = y_train[train_pool_indices]
        train_pool_labels              = tf.one_hot(train_pool_labels, self.num_classes)

        self.train_pool_init_dataset   = self.set_transform('train_pool_init', Mode.TRAIN, train_pool_indices, train_pool_images, train_pool_labels)
        self.train_pool_dataset        = self.set_transform('train_pool'     , Mode.TRAIN, train_pool_indices, train_pool_images, train_pool_labels, self.pool_batch_size)
        self.train_pool_eval_dataset   = self.set_transform('train_pool_eval', Mode.EVAL , train_pool_indices, train_pool_images, train_pool_labels)

        # validation set
        validation_indices      = self.get_all_validation_indices()
        validation_images       = X_train[validation_indices]
        validation_labels       = y_train[validation_indices]
        self.validation_dataset = self.set_transform('validation', Mode.EVAL, validation_indices, validation_images, validation_labels)

        # train_unpool_set
        train_unpool_indices           = self.get_all_unpool_train_indices()
        train_unpool_images            = X_train[train_unpool_indices]
        train_unpool_labels            = self.train_unpool_soft_labels
        self.train_unpool_dataset      = self.set_transform('train_unpool'     , Mode.EVAL, train_unpool_indices, train_unpool_images, train_unpool_labels, self.unpool_batch_size)
        self.train_unpool_eval_dataset = self.set_transform('train_unpool_eval', Mode.EVAL, train_unpool_indices, train_unpool_images, train_unpool_labels)

    def build_iterators(self):
        super(SemiSupervisedDatasetWrapper, self).build_iterators()
        self.train_pool_iterator        = self.train_pool_dataset.make_one_shot_iterator()
        self.train_pool_init_iterator   = self.train_pool_init_dataset.make_one_shot_iterator()
        self.train_pool_eval_iterator   = self.train_pool_eval_dataset.make_initializable_iterator()
        self.train_unpool_iterator      = self.train_unpool_dataset.make_initializable_iterator()
        self.train_unpool_eval_iterator = self.train_unpool_eval_dataset.make_initializable_iterator()

    def set_handles(self, sess):
        super(SemiSupervisedDatasetWrapper, self).set_handles(sess)
        self.train_pool_handle        = sess.run(self.train_pool_iterator.string_handle())
        self.train_pool_init_handle   = sess.run(self.train_pool_init_iterator.string_handle())
        self.train_pool_eval_handle   = sess.run(self.train_pool_eval_iterator.string_handle())
        self.train_unpool_handle      = sess.run(self.train_unpool_iterator.string_handle())
        self.train_unpool_eval_handle = sess.run(self.train_unpool_eval_iterator.string_handle())

    def get_handle(self, name):
        if name == 'train_pool':
            return self.train_pool_handle
        if name == 'train_pool_init':
            return self.train_pool_init_handle
        elif name == 'train_pool_eval':
            return self.train_pool_eval_handle
        elif name == 'train_unpool':
            return self.train_unpool_handle
        elif name == 'train_unpool_eval':
            return self.train_unpool_eval_handle
        return super(SemiSupervisedDatasetWrapper, self).get_handle(name)

    def print_stats(self):
        super(SemiSupervisedDatasetWrapper, self).print_stats()
        self.log.info(' UNSUPERVISED_PERCENTAGE: {}'.format(self.unsupervised_percentage))
        self.log.info(' UNSUPERVISED_PERCENTAGE_BATCH: {}'.format(self.unsupervised_percentage_batch))

    def update_soft_labels(self, new_soft_labels, step):
        """
        :param new_soft_labels: updating the tain_unpooled soft labels
        :param step: gloabl step
        :return: None
        """
        if new_soft_labels.shape != self.train_unpool_soft_labels.shape:
            err_str = 'new_soft_labels.shape does not match self.train_unpool_soft_labels.shape. ({}!={})'\
                .format(new_soft_labels.shape, self.train_unpool_soft_labels.shape)
            self.log.error(err_str)
            raise AssertionError(err_str)

        sampled_old_values = self.train_unpool_soft_labels[0:5]
        self.log.info('updating the train_unpool soft labels for global_step={}'.format(step))
        self.train_unpool_soft_labels = new_soft_labels

        debug_str = 'first 5 train unpooled soft labels:\n old_values = {}\n new_values = {}'\
            .format(sampled_old_values, self.train_unpool_soft_labels[0:5])
        self.log.info(debug_str)
        print(debug_str)
