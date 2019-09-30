""" Dividing the feature space into 100 segments. From each segment we choose the (100000/X)% of the farthest samples
    From the segment's center. X is the number of the unlabeled samples"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_TB.lib.trainers.active_trainer import ActiveTrainer
from sklearn.cluster import KMeans
import numpy as np


class FarthestKMeansTrainer(ActiveTrainer):

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

        self.log.info('for each center, find new farthest samples')
        centers  = KM.cluster_centers_
        segments = KM.predict(unlabeled_features_vec)
        counts = np.bincount(segments, minlength=100)
        clusters_dict = dict(zip(range(100), counts))  # debug
        print('clusters_dict is: {}'.format(clusters_dict))
        self.log.info('clusters_dict is: {}'.format(clusters_dict))

        selection_prob = 1000 / unlabeled_features_vec.shape[0]
        budget_dict = {}
        for segment_id in range(100):
            budget_dict[segment_id] = int(np.round(selection_prob * clusters_dict[segment_id]))

        budget_sum_pre = np.sum(budget_dict.values())
        self.log.info('budget_dict before updating is: {}\n the sum is {}'.format(budget_dict, budget_sum_pre))  # debug

        if np.sum(budget_sum_pre) == 1000:
            pass
        else:
            to_increase = budget_sum_pre < 1000
            while np.sum(budget_dict.values()) != 1000:
                max_id = max(budget_dict, key=budget_dict.get)
                self.log.info('changing the budget from segment_id={} from {}. to_increase={}'
                              .format(max_id, budget_dict[max_id], to_increase))
                if to_increase:
                    budget_dict[max_id] += 1
                else:
                    budget_dict[max_id] -= 1

        budget_sum_post = np.sum(budget_dict.values())
        self.log.info('budget_dict after updating is: {}\n the sum is {}'.format(budget_dict, budget_sum_post))  # debug
        if budget_sum_post != 1000:
            err_str = 'sum(budget_dict) equals {} instead of 1000'.format(budget_sum_post)
            self.log.error(err_str)
            raise AssertionError(err_str)

        new_indices = []
        for segment_id in range(100):
            center = centers[segment_id]
            segment_indices = np.where(segments == segment_id)[0]
            features_for_segment = unlabeled_features_vec[segment_indices]

            # debug
            if features_for_segment.shape[0] < budget_dict[segment_id]:
                err_str = 'features_for_segment (segment_id={}) has only {} elements, but budget_dict[segment_id]={}'\
                    .format(segment_id, features_for_segment.shape[0], budget_dict[segment_id])
                self.log.error(err_str)
                raise AssertionError(err_str)
            if features_for_segment.shape[0] != clusters_dict[segment_id]:
                err_str = 'features_for_segment (segment_id={}) has {} elements instead of {}' \
                    .format(segment_id, features_for_segment.shape[0], clusters_dict[segment_id])
                self.log.error(err_str)
                raise AssertionError(err_str)

            diff_vec = features_for_segment - center
            u_vec = np.linalg.norm(diff_vec, ord=1, axis=1)

            self.log.info('Finding the farthest samples from the center of segment_id={}'.format(segment_id))
            if budget_dict[segment_id] > 0:
                farthest_segment_indices = u_vec.argsort()[-budget_dict[segment_id]:]
                farthest_indices = segment_indices[farthest_segment_indices]
                new_indices_tmp =  [unlabeled_vec_dict.values()[i] for i in farthest_indices]
                new_indices += new_indices_tmp
                if len(new_indices_tmp) != budget_dict[segment_id]:
                    err_str = 'for segment_id={} len(new_indices_tmp) equals {} instead of budget_dict[segment_id]={}'\
                        .format(segment_id, len(new_indices_tmp), budget_dict[segment_id])
                    self.log.error(err_str)
                    raise AssertionError(err_str)
            else:
                self.log.info('for segment_id={} no new samples are chosen because budget_dict[segment_id] = 0'.format(segment_id))

        return new_indices

# # debug
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
# counts = np.bincount(segments, minlength=100)
# clusters_dict = dict(zip(range(100), counts)) # debug
# print('clusters_dict is: {}'.format(clusters_dict))
#
# selection_prob = 1000 / unlabeled_features_vec.shape[0]
# budget_dict={}
# for segment_id in range(100):
#     budget_dict[segment_id] = int(np.round(selection_prob * clusters_dict[segment_id]))
#
# if np.sum(budget_dict.values()) == 1000:
#     pass
# else:
#     to_increase = np.sum(budget_dict.values()) < 1000
#     while np.sum(budget_dict.values()) != 1000:
#         max_id = max(budget_dict, key=budget_dict.get)
#         print('changing the budget from segment_id={} from {}. to_increase={}'.format(max_id, budget_dict[max_id], to_increase))
#         if to_increase:
#             budget_dict[max_id] += 1
#         else:
#             budget_dict[max_id] -= 1
#
# new_indices = []
# for segment_id in range(100):
#     center = centers[segment_id]
#     segment_indices = np.where(segments == segment_id)[0]
#     features_for_segment = unlabeled_features_vec[segment_indices]
#
#     # debug
#     if features_for_segment.shape[0] < budget_dict[segment_id]:
#         err_str = 'features_for_segment (segment_id={}) has only {} elements, but budget_dict[segment_id]={}'\
#             .format(segment_id, features_for_segment.shape[0], budget_dict[segment_id])
#         raise AssertionError(err_str)
#     if features_for_segment.shape[0] != clusters_dict[segment_id]:
#         err_str = 'features_for_segment (segment_id={}) has {} elements instead of {}' \
#             .format(segment_id, features_for_segment.shape[0], clusters_dict[segment_id])
#         raise AssertionError(err_str)
#
#     diff_vec = features_for_segment - center
#     u_vec = np.linalg.norm(diff_vec, ord=1, axis=1)
#
#     farthest_segment_indices = u_vec.argsort()[-budget_dict[segment_id]:]
#     farthest_indices = segment_indices[farthest_segment_indices]
#     new_indices_tmp = [unlabeled_vec_dict.values()[i] for i in farthest_indices]
#     new_indices += new_indices_tmp
#
# new_indices

