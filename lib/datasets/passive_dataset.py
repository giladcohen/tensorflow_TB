import numpy as np
from lib.datasets.dataset import DataSet

class PassiveDataSet(DataSet):
    def __init__(self, *args, **kwargs):
        super(PassiveDataSet, self).__init__(*args, **kwargs)

        self.cap = self.prm.dataset.CAP

        if self.cap is None:
            self.log.warning(self.__str__() + ' was initialized with CAP=None. Setting CAP=size ({})'.format(self.size))
            self.cap = self.size
        self.clusters = self.prm.dataset.CLUSTERS

        self.pool = []  # starting from zero indices in the pool
        self.available_samples = range(self.size)

        self.initialize_pool()

    def print_stats(self):
        super(PassiveDataSet, self).print_stats()
        self.log.info(self.__str__() + ': CAP: {}'.format(self.cap))
        self.log.info(self.__str__() + ': CLUSTERS: {}'.format(self.clusters))

    def initialize_pool(self):
        indices = np.random.choice(self.available_samples, self.cap, replace=False)
        indices = indices.tolist()
        self.update_pool_with_indices(indices)

    def update_pool_with_indices(self, indices):
        """indices must be of type list"""
        self.pool += indices
        self.pool = sorted(self.pool)
        self.available_samples = [i for j, i in enumerate(self.available_samples) if i not in self.pool]
        self.log.info('updating pool length to {}'.format(self.pool_size()))
        if self.pool_size() > self.cap:
            err_str = 'update_pool_with_indices: pool size ({}) surpassed cap ({})'.format(self.pool_size(), self.cap)
            self.log.error(err_str)
            raise AssertionError(err_str)

