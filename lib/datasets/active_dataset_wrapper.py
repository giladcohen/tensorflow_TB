from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from lib.datasets.dataset_wrapper import DatasetWrapper
from utils.enums import Mode

class ActiveDatasetWrapper(DatasetWrapper):
    def __init__(self, *args, **kwargs):
        super(ActiveDatasetWrapper, self).__init__(*args, **kwargs)
        self.clusters  = self.prm.dataset.CLUSTERS
        self.init_size = self.prm.dataset.INIT_SIZE
        self.cap       = self.prm.dataset.CAP

        self.train_pool_dataset       = None
        self.train_pool_eval_dataset  = None

        self.train_pool_iterator      = None
        self.train_pool_eval_iterator = None

        self.train_pool_handle        = None
        self.train_pool_eval_handle   = None

    def set_data_info(self):
        """
        There is no ref to load data info from. Therefore we need to construct a dataset with pool size of init_size
        """
        # We start by setting self.train_validation_info as parent
        super(ActiveDatasetWrapper, self).set_data_info()

        # from all the train samples, chose only init_size samples
        train_indices = self.get_all_train_indices()

        # now chose init_size indices to add to the pool
        train_pool_indices = self.rand_gen.choice(train_indices, self.init_size, replace=False)
        train_pool_indices = train_pool_indices.tolist()
        train_pool_indices.sort()
        for sample in self.train_validation_info:
            if sample['index'] in train_pool_indices:
                sample['in_pool'] = True

    def get_all_train_indices(self):
        """
        :return: all train indices, regardless of 'in_pool' values
        """
        indices = []
        for sample in self.train_validation_info:
            if sample['dataset'] == 'train':
                indices.append(sample['index'])
        indices.sort()
        return indices

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
        super(ActiveDatasetWrapper, self).set_datasets(X_train, y_train, X_test, y_test)

        # train_pool_set
        train_pool_indices           = [element['index'] for element in self.train_validation_info if element['dataset'] == 'train' and element['in_pool']]
        train_pool_images            = X_train[train_pool_indices]
        train_pool_labels            = y_train[train_pool_indices]
        self.train_pool_dataset      = self.set_transform('train_pool', Mode.TRAIN, train_pool_images, train_pool_labels)
        self.train_pool_eval_dataset = self.set_transform('train_pool', Mode.EVAL , train_pool_images, train_pool_labels)

    def build_iterators(self):
        super(ActiveDatasetWrapper, self).build_iterators()
        self.train_pool_iterator      = self.train_pool_dataset.make_one_shot_iterator()
        self.train_pool_eval_iterator = self.train_pool_eval_dataset.make_initializable_iterator()

    def set_handles(self, sess):
        super(ActiveDatasetWrapper, self).set_handles(sess)
        self.train_pool_handle      = sess.run(self.train_pool_iterator.string_handle())
        self.train_pool_eval_handle = sess.run(self.train_pool_eval_iterator.string_handle())

    def get_handle(self, name):
        if name == 'train_pool':
            return self.train_pool_handle
        elif name == 'train_pool_eval':
            return self.train_pool_eval_handle
        return super(ActiveDatasetWrapper, self).get_handle(name)

    def print_stats(self):
        super(ActiveDatasetWrapper, self).print_stats()
        self.log.info(' CLUSTERS: {}'.format(self.clusters))
        self.log.info(' INIT_SIZE: {}'.format(self.init_size))
        self.log.info(' CAP: {}'.format(self.cap))

    def update_pool(self, clusters=None, indices=None):
        """
        updating the train_pool indices with #clusters of samples, or with specific indices
        :param clusters: integer, number of new samples to add to train_pool
        :param indices: list of indices to add to train_pool
        :return: None
        """
        if clusters is None:
            clusters = self.clusters
        if indices is None:
            # indices are not provided - selecting list of indices
            unpool_train_indices = self.get_all_unpool_train_indices()
            if len(unpool_train_indices) < clusters:
                self.log.warning('Adding {} indices instead of {} to pool. pool is full'.format(len(unpool_train_indices), clusters))
                indices = unpool_train_indices
            else:
                indices = self.rand_gen.choice(unpool_train_indices, clusters, replace=False)
                indices = indices.tolist()
                indices.sort()
        self.update_pool_with_indices(indices)
        self.save_data_info()
        self.build_datasets()
        self.build_iterators()

    def update_pool_with_indices(self, indices):
        """Updating train_pool dataset with new indices
        :param indices: list of new indices(int)
        :return None
        """
        self.assert_unique_indices(indices)  # time consuming.
        for sample in self.train_validation_info:
            if sample['index'] in indices:
                sample['in_pool'] = True

        self.log.info('updated train_pool length to {}'.format(self.pool_size))
        if self.pool_size > self.cap:
            err_str = 'update_pool_with_indices: pool size ({}) surpassed cap ({})'.format(self.pool_size, self.cap)
            self.log.error(err_str)
            raise AssertionError(err_str)

    @property
    def pool_size(self):
        return len(self.get_all_pool_train_indices())

    def assert_unique_indices(self, indices):
        """Asserting that the new indices to add to train_pool are not already in train_pool
        :param indices: new indices to add to train_pool dataset
        :return: None
        """
        if not set(self.get_all_pool_train_indices()).isdisjoint(indices):
            err_str = 'assert_unique_indices: some index/indices are already in pool.\nindices={}'.format(indices)
            self.log.error(err_str)
            raise AssertionError(err_str)

    def save_data_info(self, info_save_path=None):
        """Saving self.train_validation_info into disk"""
        save_file = 'train_validation_info_' + 'lp_' + str(self.pool_size)
        info_save_path = os.path.join(self.prm.train.train_control.ROOT_DIR, save_file)
        super(ActiveDatasetWrapper, self).save_data_info(info_save_path)
