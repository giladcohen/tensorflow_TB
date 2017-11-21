from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer_base import ActiveTrainerBase
from sklearn.cluster import KMeans
import numpy as np


class FarthestKMeansTrainer(ActiveTrainerBase):

    def select_new_samples(self):

        # prediction
        features_vec, _ = self.collect_features('train')

        labeled_features_vec      = features_vec[self.dataset.train_dataset.pool]
        unlabeled_features_vec    = features_vec[self.dataset.train_dataset.available_samples]

        labeled_vec_dict   = dict(zip(range(labeled_features_vec.shape[0])  , self.dataset.train_dataset.pool))
        unlabeled_vec_dict = dict(zip(range(unlabeled_features_vec.shape[0]), self.dataset.train_dataset.available_samples))

        self.log.info('performing K-Means for the labeled train features. K=100')
        KM = KMeans(n_clusters=100,
                    n_init=10,
                    random_state=self.rand_gen,
                    n_jobs=10)
        KM.fit(labeled_features_vec)

        self.log.info('for each center, find 10 new farthest samples')
        centers  = KM.cluster_centers_
        segments = KM.predict(unlabeled_features_vec)
        unique, counts = np.unique(segments, return_counts=True)  # debug
        clusters_dict = dict(zip(unique, counts))                 # debug
        assert len(unique) == 100, 'the length of unique must equal 100 (number of segments'

        new_indices = []
        for segment_id in range(100):
            center = centers[segment_id]
            segment_indices             = np.where(segments == segment_id)[0]
            features_for_segment        = unlabeled_features_vec[segment_indices]

            # debug
            if features_for_segment.shape[0] < 10:
                err_str = 'unlabeled_features_for_segment (segment_id={}) has less than {} elements'.format(segment_id, 10)
                self.log.error(err_str)
                raise AssertionError(err_str)
            if features_for_segment.shape[0] != clusters_dict[segment_id]:
                err_str = 'unlabeled_features_for_segment (segment_id={}) has {} elements instead of {}' \
                                         .format(segment_id, features_for_segment.shape[0], clusters_dict[segment_id])
                self.log.error(err_str)
                raise AssertionError(err_str)

            diff_vec = features_for_segment - center
            u_vec = np.linalg.norm(diff_vec, ord=1, axis=1)

            self.log.info('Finding the 10 farthest samples from the center of segment_id={}'.format(segment_id))
            farthest_segment_indices = u_vec.argsort()[-10:]
            farthest_indices = segment_indices[farthest_segment_indices]
            new_indices_tmp =  [unlabeled_vec_dict.values()[i] for i in farthest_indices]
            new_indices += new_indices_tmp

        return new_indices

# debug
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# from lib.trainers.active_trainer_base import ActiveTrainerBase
# from sklearn.cluster import KMeans
# import numpy as np
#
# features_vec = np.random.uniform(size=(50000, 2))
# available_samples = range(50000)
# pool = np.random.choice(available_samples, 1000, replace=False)
# pool = sorted(pool)
# available_samples = [i for j, i in enumerate(available_samples) if i not in pool]
#
# labeled_features_vec = features_vec[pool]
# unlabeled_features_vec = features_vec[available_samples]
#
# labeled_vec_dict = dict(zip(range(labeled_features_vec.shape[0]), pool))
# unlabeled_vec_dict = dict(zip(range(unlabeled_features_vec.shape[0]), available_samples))
#
# KM = KMeans(n_clusters=100,
#             n_init=1,
#             random_state=None,
#             n_jobs=1)
# KM.fit(labeled_features_vec)
# centers = KM.cluster_centers_
# segments = KM.predict(unlabeled_features_vec)
# unique, counts = np.unique(segments, return_counts=True)  # debug
# clusters_dict = dict(zip(unique, counts))                 # debug
# assert len(unique) == 100, 'the length of unique must equal 100 (number of segments'
#
# new_indices = []
# for segment_id in range(100):
#     center = centers[segment_id]
#     segment_indices = np.where(segments == segment_id)[0]
#     features_for_segment = unlabeled_features_vec[segment_indices]
#
#     # debug
#     if features_for_segment.shape[0] < 10:
#         err_str = 'unlabeled_features_for_segment (segment_id={}) has less than {} elements'.format(segment_id, 10)
#         raise AssertionError(err_str)
#     if features_for_segment.shape[0] != clusters_dict[segment_id]:
#         err_str = 'unlabeled_features_for_segment (segment_id={}) has {} elements instead of {}' \
#             .format(segment_id, features_for_segment.shape[0], clusters_dict[segment_id])
#         raise AssertionError(err_str)
#
#     diff_vec = features_for_segment - center
#     u_vec = np.linalg.norm(diff_vec, ord=1, axis=1)
#
#     farthest_segment_indices = u_vec.argsort()[-10:]
#     farthest_indices = segment_indices[farthest_segment_indices]
#     new_indices_tmp = [unlabeled_vec_dict.values()[i] for i in farthest_indices]
#     new_indices += new_indices_tmp
#
# new_indices
#
