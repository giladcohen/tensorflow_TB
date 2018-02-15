from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer import ActiveTrainer
import numpy as np


class MostUncertainedTrainer(ActiveTrainer):
    def select_new_samples(self):

        # analyzing (evaluation)
        _, predictions_vec = self.collect_features('train')

        unlabeled_predictions_vec = predictions_vec[self.dataset.train_dataset.available_samples]
        unlabeled_vec_dict = dict(zip(range(unlabeled_predictions_vec.shape[0]), self.dataset.train_dataset.available_samples))

        #prediction
        self.log.info('Calculating the uncertainty score vector')
        u_vec = self.uncertainty_score(unlabeled_predictions_vec)
        unlabeled_predictions_indices = u_vec.argsort()[-self.dataset.train_dataset.clusters:]
        new_indices = [unlabeled_vec_dict.values()[i] for i in unlabeled_predictions_indices]

        return new_indices

    def uncertainty_score(self, sf_arr):
        """
        Calculates the uncertainty score for sf_arr
        :param sf_arr: np.float32 array of all the predictions of the network
        :return: uncertainty score for every vector
        """
        return np.power(sf_arr.max(axis=1), -1)
