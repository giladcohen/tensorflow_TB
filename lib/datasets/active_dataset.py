import numpy as np
from lib.datasets.passive_dataset import PassiveDataSet

class ActiveDataSet(PassiveDataSet):

    def initialize_pool(self):
        while self.pool_size() < self.batch_size:
            self.log.info('Small pool size: {}. Adding to pool {} random elements'.format(self.pool_size(), self.clusters))
            self.update_pool()

    def update_pool(self, clusters=None, indices=None):
        """Indices must be None or list"""
        if indices is not None:
            return self.update_pool_with_indices(indices)
        if clusters is None:
            clusters = self.clusters
        if len(self.available_samples) < clusters:
            indices = self.available_samples
            self.log.warning('Adding {} indices instead of {} to pool. pool is full'.format(len(indices), clusters))
        else:
            indices = np.random.choice(self.available_samples, clusters, replace=False)
            indices = indices.tolist()
        self.update_pool_with_indices(indices)

    def update_pool_with_indices(self, indices):
        """indices must be list type"""
        self.assert_unique_indices(indices)  # time consuming. There is no sense checking it for passive dataset.
        super(ActiveDataSet, self).update_pool_with_indices(indices)

    def assert_unique_indices(self, indices):
        for index in indices:
            if index in self.pool:
                err_str = 'update_pool_with_indices: index {} is already in pool.'.format(index)
            if index not in self.available_samples:
                err_str = 'update_pool_with_indices: index {} is not in available_samples.'.format(index)
            if 'err_str' in locals():
                self.log.error(err_str)
                raise AssertionError(err_str)

