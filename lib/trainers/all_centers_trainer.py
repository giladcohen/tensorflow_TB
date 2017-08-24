from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer_base import ActiveTrainerBase
from lib.all_centers_kmeans import AllCentersKMeans
from sklearn.neighbors import NearestNeighbors


class AllCentersTrainer(ActiveTrainerBase):
    def select_new_samples(self):
        lp = self.dataset.train_dataset.pool_size()

        # analyzing (evaluation)
        features_vec = self.collect_features('train')

        # prediction
        KM = AllCentersKMeans(name='AllCentersKMean', prm=self.prm,
                              fixed_centers=features_vec[self.dataset.train_dataset.pool],
                              n_clusters=lp + self.clusters,
                              random_state=self.rand_gen)
        KM.fit(features_vec)
        centers = KM.cluster_centers_
        new_centers = centers[lp:(lp + self.clusters)]
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(features_vec)
        indices = nbrs.kneighbors(new_centers, return_distance=False)  # get indices of NNs of new centers
        indices = indices.T[0].tolist()

        # exclude existing labels in pool
        already_pooled_cnt = 0  # number of indices of samples that we added to pool already
        for myItem in indices:
            if myItem in self.dataset.train_dataset.pool:
                already_pooled_cnt += 1
                indices.remove(myItem)
                self.log.info('Removing value {} from indices because it already exists in pool'.format(myItem))
        self.log.info('{} indices were already in pool. Randomized indices will be chosen instead of them'.format(
            already_pooled_cnt))
        self.dataset.train_dataset.update_pool(indices=indices)
        self.dataset.train_dataset.update_pool(clusters=already_pooled_cnt)
