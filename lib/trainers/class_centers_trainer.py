from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer_base import ActiveTrainerBase
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np


class ClassCentersTrainer(ActiveTrainerBase):
    def select_new_samples(self):
        num_classes = self.model.num_classes
        delta = self.dataset.train_dataset.clusters // num_classes

        # analyzing (evaluation)
        features_vec = self.collect_features('train')
        labeled_features_vec   = features_vec[self.dataset.train_dataset.pool]
        labeled_features_vec_dict = dict(zip(range(labeled_features_vec.shape[0]), self.dataset.train_dataset.pool))
        unlabeled_features_vec = features_vec[self.dataset.train_dataset.available_samples]
        unlabeled_features_vec_dict = dict(zip(range(unlabeled_features_vec.shape[0]), self.dataset.train_dataset.available_samples))

        self.log.info('building kNN space only for the unlabeled train features')
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(unlabeled_features_vec)

        # prediction
        self.log.info('performing K-Means only for the labeled train features. K={}'.format(num_classes))
        KM = KMeans(n_clusters=num_classes,
                    n_init=100,
                    random_state=self.rand_gen,
                    n_jobs=10)
        KM.fit(labeled_features_vec)

        self.log.info('for each center, find {} new K-MEANS using only the unlabeled data that correspond to it'.format(delta))
        estimated_labels = KM.predict(unlabeled_features_vec)
        unique, counts = np.unique(estimated_labels, return_counts=True)
        clusters_dict = dict(zip(unique, counts))

        new_indices = []
        for cluster_id in range(num_classes):
            unlabeled_features_for_cluster = unlabeled_features_vec[estimated_labels == cluster_id]

            # debug
            if unlabeled_features_for_cluster.shape[0] < delta:
                err_str = 'unlabeled_features_for_cluster (cluster_id={}) has less than {} elements'.format(delta)
                self.log.error(err_str)
                raise AssertionError(err_str)
            if unlabeled_features_for_cluster.shape[0] != clusters_dict[cluster_id]:
                err_str = 'unlabeled_features_for_cluster (cluster_id={}) has {} elements instead of {}' \
                                         .format(cluster_id, unlabeled_features_for_cluster.shape[0], clusters_dict[cluster_id])
                self.log.error(err_str)
                raise AssertionError(err_str)

            self.log.info('building K-Means model for cluster {} and finding its sub clusters'.format(cluster_id))
            KM = KMeans(n_clusters=delta,
                        n_init=100,
                        random_state=self.rand_gen,
                        n_jobs=10)
            KM.fit(unlabeled_features_for_cluster)
            sub_centers = KM.cluster_centers_
            unlabeled_indices = nbrs.kneighbors(sub_centers, return_distance=False)  # get indices of NNs of new centers
            unlabeled_indices = unlabeled_indices.T[0].tolist()
            new_indices = new_indices + [unlabeled_features_vec_dict.values()[i] for i in unlabeled_indices]

        return new_indices
