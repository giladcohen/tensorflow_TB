from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_TB.lib.trainers.active_trainer import ActiveTrainer
import numpy as np

class FarthestUncertainedSamplesTrainer(ActiveTrainer):
    def select_new_samples(self):

        # analyzing (evaluation)
        features_vec, predictions_vec = self.collect_features('train')

        unlabeled_features_vec = features_vec[self.dataset.train_dataset.available_samples]
        unlabeled_predictions_vec = predictions_vec[self.dataset.train_dataset.available_samples]
        unlabeled_vec_dict = dict(zip(range(unlabeled_predictions_vec.shape[0]), self.dataset.train_dataset.available_samples))

        #prediction
        self.log.info('Calculating the uncertainy score vector')
        u_vec = self.uncertainty_score(unlabeled_predictions_vec)

        # choosing the 10k most uncertained unlabeled samples and collecting their features
        unlabeled_top_10k_indices_tmp = u_vec.argsort()[-10000:]  # TODO(gilad): Support modifying this number
        unlabeled_top_10k_indices = [unlabeled_vec_dict.values()[i] for i in unlabeled_top_10k_indices_tmp]
        unlabeled_top_10k = features_vec[unlabeled_top_10k_indices]

        # building 10000 x 10000 matrix of distances
        self.log.info('Building 10kx10k matrix of distances')
        distance_mat = np.empty(shape=[10000, 10000], dtype=np.float32)
        for i in xrange(10000):
            for j in xrange(10000):
                distance_mat[i, j] = np.linalg.norm(unlabeled_top_10k[i] - unlabeled_top_10k[j], ord=1)

        distance_vec = distance_mat.sum(axis=1)         # summing all columns
        new_indices_tmp = distance_vec.argsort()[-1000:].tolist()
        new_indices = [unlabeled_top_10k_indices[i] for i in new_indices_tmp]

        return new_indices

    def uncertainty_score(self, sf_arr):
        """
        Calculates the uncertainty score for sf_arr
        :param sf_arr: np.float32 array of all the predictions of the network
        :return: uncertainty score for every vector
        """
        return np.power(sf_arr.max(axis=1), -1)
