import numpy as np
from lib.datasets.dataset import DataSet

class ActiveDataSet(DataSet):

    def __init__(self, *args, **kwargs):
        super(ActiveDataSet, self).__init__(*args, **kwargs)
        self.init_size = self.prm.dataset.INIT_SIZE
        self.clusters  = self.prm.dataset.CLUSTERS
        self.cap       = self.prm.dataset.CAP  # must not be None
        self.assert_config()

    def assert_config(self):
        if self.cap is None:
            err_str = self.__str__() + ': CAP cannot be None'
            self.log.error(err_str)
            raise AssertionError(err_str)
        if self.init_size is None:
            self.log.warning(self.__str__() + 'Initialized with INIT_SIZE=None. Setting INIT_SIZE=CAP ({})'.format(self.cap))
            self.init_size = self.cap

    def initialize_pool(self):
        self.log.info('Initializing pool with {} random values'.format(self.init_size))
        self.pool = []
        self.available_samples = range(self.size)
        self.update_pool(clusters=self.init_size)

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

    def assert_unique_indices(self, indices):
        for index in indices:
            if index in self.pool:
                err_str = 'update_pool_with_indices: index {} is already in pool.'.format(index)
            if index not in self.available_samples:
                err_str = 'update_pool_with_indices: index {} is not in available_samples.'.format(index)
            if 'err_str' in locals():
                self.log.error(err_str)
                raise AssertionError(err_str)

    def update_pool_with_indices(self, indices):
        """indices must be of type list"""
        self.assert_unique_indices(indices)  # time consuming.
        self.pool += indices
        self.pool = sorted(self.pool)
        self.available_samples = [i for j, i in enumerate(self.available_samples) if i not in self.pool]
        self.minibatch_server.set_pool(self.pool)
        self.log.info('updated pool length to {}'.format(self.pool_size()))
        if self.pool_size() > self.cap:
            err_str = 'update_pool_with_indices: pool size ({}) surpassed cap ({})'.format(self.pool_size(), self.cap)
            self.log.error(err_str)
            raise AssertionError(err_str)
