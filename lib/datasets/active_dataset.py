import numpy as np
from lib.datasets.passive_dataset import PassiveDataSet

class ActiveDataSet(PassiveDataSet):
    def __init__(self, *args, **kwargs):
        super(ActiveDataSet, self).__init__(*args, **kwargs)

        self.n_clusters = self.prm.dataset.N_CLUSTERS

        self.initialize_pool()

    def print_stats(self):
        super(ActiveDataSet, self).print_stats()
        self.log.info(self.__str__() + ': N_CLUSTERS: {}'.format(self.n_clusters))

    def initialize_pool(self):
        while self.pool_size() < self.batch_size:
            self.log.info('Small pool size: {}. Adding to pool {} random elements'.format(self.pool_size(), self.n_clusters))
            self.update_pool()

    def update_pool(self, n_clusters=None, indices=None):
        if indices is not None:
            return self.update_pool_with_indices(indices)
        if n_clusters is None:
            n_clusters = self.n_clusters
        if len(self.available_samples) < n_clusters:
            indices = self.available_samples
            self.log.warning('Adding {} indices instead of {} to pool. pool is full'.format(len(indices), n_clusters))
        else:
            indices = np.random.choice(self.available_samples, n_clusters, replace=False)
        self.update_pool_with_indices(indices)

    def update_pool_with_indices(self, indices):
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

