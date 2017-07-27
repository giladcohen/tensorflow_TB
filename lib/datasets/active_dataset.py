import numpy as np
from lib.datasets.dataset import DataSet

class ActiveDataSet(DataSet):
    def __init__(self, *args, **kwargs):
        super(DataSet, self).__init__(*args, **kwargs)

        self.n_clusters = self.prm.dataset.N_CLUSTERS
        self.cap        = self.prm.dataset.CAP

        self.pool = []  # starting from zero indices in the pool
        self.available_samples = range(self.size)

        self.initialize_pool()

    def print_stats(self):
        super(ActiveDataSet, self).print_stats()
        self.log.info(self.__str__() + ': N_CLUSTERS: {}'.format(self.n_clusters))
        self.log.info(self.__str__() + ': CAP: {}'.format(self.cap))

    def initialize_pool(self):
        while len(self.pool) < self.batch_size:
            self.log.info('Small pool length: {}. Adding to pool {} random elements'.format(len(self.pool), self.n_clusters))
            self.update_pool_rand()

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
        for index in indices:
            if index in self.pool:
                err_str = 'update_pool_with_indices: index {} is already in pool.'.format(index)
            if index not in self.available_samples:
                err_str = 'update_pool_with_indices: index {} is not in available_samples.'.format(index)
            if 'err_str' in locals():
                self.log.error(err_str)
                raise AssertionError(err_str)
        self.pool += indices
        self.pool = sorted(self.pool)
        self.available_samples = [i for j, i in enumerate(self.available_samples) if i not in self.pool]
        self.log.info('updating pool length to {}'.format(len(self.pool)))
