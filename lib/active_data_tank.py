from lib.data_tank import DataTank
import random
import numpy as np


class ActiveDataTank(DataTank):

    def __init__(self, n_clusters, *args, **kwargs):
        super(ActiveDataTank, self).__init__(*args, **kwargs)
        self.pool = []
        self.available_samples = range(self.N)
        self.n_clusters = n_clusters
        self.initialize_pool()

    def initialize_pool(self):
        while len(self.pool) < self.batch_size:
            print('Small pool length (%0d). Adding to pool %0d random elements' %(len(self.pool), self.n_clusters))
            self.update_pool_rand()

    def update_pool_rand(self, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.n_clusters
        if len(self.available_samples) < n_clusters:
            indices = self.available_samples
            print ('Adding %0d indices instead of %d to pool. pool is full' % (len(indices), n_clusters))
        else:
            indices = random.sample(self.available_samples, n_clusters)
        self._update_pool_common(indices)

    def update_pool(self, indices):
        ''' Updating pool with fixed indices '''
        li  = len(indices)
        lai = len(self.available_samples)
        if lai < li:
            raise ValueError('Calling update_pool with %0d indices and we have %0d available samples' %(li, lai))
        assert set(indices).isdisjoint(self.pool)
        self._update_pool_common(indices)

    def _update_pool_common(self, indices):
        self.pool += indices
        self.pool = sorted(self.pool)
        self.available_samples = [i for j, i in enumerate(self.available_samples) if i not in self.pool]
        print('updating pool length to %0d' % len(self.pool))
