from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer_base import ActiveTrainerBase
import numpy as np


class MostUncertainedBalancedTrainer(ActiveTrainerBase):

    def __init__(self, *args, **kwargs):
        super(MostUncertainedBalancedTrainer, self).__init__(*args, **kwargs)
        self.num_classes = self.model.num_classes

        # creating a numpy matrix that partitions the labels with different rows
        self.class_samples = np.empty(shape=[self.num_classes, int(self.dataset.train_dataset.size / self.num_classes)])
        for cls in xrange(self.num_classes):
            self.class_samples[cls] = np.argwhere(self.dataset.train_dataset.labels == cls)

    def train_step(self):
        '''Implementing one training step'''
        lp = self.dataset.train_dataset.pool_size()
        if lp > 0:
            super(MostUncertainedBalancedTrainer, self).train_step()
        else:
            self.log.info('Pool is empty. Filling it with balanced dataset')
            self.select_init_samples()

    def select_init_samples(self):
        tmp_pool = []
        for cls in xrange(self.num_classes):
            tmp_pool += self.rand_gen.choice(self.class_samples[cls], int(self.clusters / self.num_classes), replace=False)
        self.add_new_samples(tmp_pool)

    def select_new_samples(self):
        class_counter = int(self.clusters / self.num_classes) * np.ones(self.num_classes, dtype=np.int32)

        # analyzing (evaluation)
        _, predictions_vec = self.collect_features('train')

        unlabeled_predictions_vec = predictions_vec[self.dataset.train_dataset.available_samples]
        unlabeled_predictions_vec_dict = dict(zip(range(unlabeled_predictions_vec.shape[0]), self.dataset.train_dataset.available_samples))

        # prediction
        self.log.info('Calculating the uncertainy score vector')
        u_vec = self.uncertainty_score(unlabeled_predictions_vec)
        unlabeled_predictions_indices = u_vec.argsort()

        # selection of new indices
        new_indices = []
        i = -1
        cnt = 0  # for debug
        while i >= -1 * len(self.dataset.train_dataset.available_samples):
            candidate_index = unlabeled_predictions_vec_dict[unlabeled_predictions_indices[i]]
            label = self.dataset.train_dataset.labels[candidate_index]
            if class_counter[label] > 0:
                new_indices += candidate_index
                class_counter[label] -= 1
                cnt += 1
            i -= 1
        assert cnt == self.clusters, 'updating pool with {} indices instead of {}'.format(cnt, self.clusters)
        return new_indices

    def uncertainty_score(self, sf_arr):
        """
        Calculates the uncertainty score for sf_arr
        :param sf_arr: np.float32 array of all the predictions of the network
        :return: uncertainty score for every vector
        """
        return np.power(sf_arr.max(axis=1), -1)
