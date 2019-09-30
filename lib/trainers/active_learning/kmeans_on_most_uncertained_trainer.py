from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_TB.lib.trainers.active_trainer import ActiveTrainer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class KMeansOnMostUncertainedTrainer(ActiveTrainer):
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
        # unlabeled_top_10k_dict = dict(zip(range(unlabeled_top_10k.shape[0]), unlabeled_top_10k_indices))

        # using KMeans to choose 1k centers from the above 10k unlabeled samples
        self.log.info('building K-Means model for the 10k most uncertained unlabeled samples')
        KM = KMeans(n_clusters=self.dataset.train_dataset.clusters,
                    n_init=10,
                    random_state=self.rand_gen,
                    n_jobs=10)
        KM.fit(unlabeled_top_10k)
        centers = KM.cluster_centers_

        # what are the nearest neighbors fot these 1000 centers out of all the unlabeled samples?
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(unlabeled_features_vec)
        unlabeled_indices = nbrs.kneighbors(centers, return_distance=False)
        unlabeled_indices = unlabeled_indices.T[0].tolist()
        new_indices = [unlabeled_vec_dict.values()[i] for i in unlabeled_indices]

        return new_indices

    def uncertainty_score(self, sf_arr):
        """
        Calculates the uncertainty score for sf_arr
        :param sf_arr: np.float32 array of all the predictions of the network
        :return: uncertainty score for every vector
        """
        return np.power(sf_arr.max(axis=1), -1)
