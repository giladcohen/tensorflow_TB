import numpy as np
from lib.base.agent_base import AgentBase

class MiniBatchServer(AgentBase):

    def __init__(self, name, prm):
        super(MiniBatchServer, self).__init__(name)
        self.prm = prm

        self.rand_gen = np.random.RandomState(self.prm.SUPERSEED)
        self.pool = None  # needs to be set manually
        self._n_elements = None  # needs to be calculated when a new pool is set.
        self._permuted_indices = None  # needs to be shuffled at new epochs and when setting new pool
        self._n_step = 0
        self._n_epoch = 0
        self._is_end_of_epoch = True
        self._current_index = 0  # total samples that were chosen in the current epoch

    def __str__(self):
        msg = 'name: {}. pool_size: {}. n_elements: {}. n_step: {}. n_epoch: {}. is_end_of_epoch: {}. current_index: {}'.format(
            self.name, len(self.pool), self._n_elements, self._n_step, self._n_epoch, self._is_end_of_epoch, self._current_index)
        return msg

    def set_pool(self, pool):
        self.log.info('Setting new pool to server. pool size {}'.format(len(pool)))
        self.pool = pool
        self._n_elements = len(self.pool)
        self._permuted_indices = self.rand_gen.permutation(self.pool)

    def get_mini_batch(self, batch_size):
        self._n_step += 1
        end = self._current_index + batch_size
        if self._is_end_of_epoch:  # previous call to get_mini_batch reached the end of the epoch
            # new epoch
            self._permuted_indices = self.rand_gen.permutation(self.pool)
            self._is_end_of_epoch = False
            self._n_epoch += 1
            self._current_index = 0
            end = batch_size
        elif end >= self._n_elements:  # current call reaches the end of the epoch
            self.log.debug('Server reached end of epoch {}'.format(self._n_epoch))
            self._current_index = self._n_elements - batch_size
            end = self._n_elements
            self._is_end_of_epoch = True

        minibatch = self._permuted_indices[self._current_index:end]
        self._current_index += batch_size
        return minibatch
