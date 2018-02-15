from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer import ActiveTrainer
from lib.all_centers_kmeans import AllCentersKMeans
from sklearn.neighbors import NearestNeighbors


class AllCentersTrainer(ActiveTrainer):
    def select_new_samples(self):
        lp = self.dataset.train_dataset.pool_size()

        # analyzing (evaluation)
        features_vec, _ = self.collect_features('train')
        labeled_features_vec   = features_vec[self.dataset.train_dataset.pool]
        labeled_features_vec_dict = dict(zip(range(labeled_features_vec.shape[0]), self.dataset.train_dataset.pool))
        unlabeled_features_vec = features_vec[self.dataset.train_dataset.available_samples]
        unlabeled_features_vec_dict = dict(zip(range(unlabeled_features_vec.shape[0]), self.dataset.train_dataset.available_samples))

        self.log.info('building kNN space only for the unlabeled train features')
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(unlabeled_features_vec)

        # prediction
        self.log.info('performing K-Means for all train features. K={}'.format(lp + self.clusters))
        KM = AllCentersKMeans(name='AllCentersKMean', prm=self.prm,
                              fixed_centers=features_vec[self.dataset.train_dataset.pool],
                              n_clusters=lp + self.clusters,
                              random_state=self.rand_gen)
        KM.fit(features_vec)
        centers = KM.cluster_centers_
        new_centers = centers[lp:(lp + self.clusters)]
        unlabeled_indices = nbrs.kneighbors(new_centers, return_distance=False)  # get indices of NNs of new centers
        unlabeled_indices = unlabeled_indices.T[0].tolist()
        new_indices = [unlabeled_features_vec_dict.values()[i] for i in unlabeled_indices]

        return new_indices
