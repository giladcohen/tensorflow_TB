"""
This trainer is like KMeansSegmentsTrainer except that here we don't choose 10 new samples out of every
one of the 1000 segments. Instead, we do averaging - adding more labels to segments that holds many samples.
Review also FarthestKMeansTrainer for reference.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer_base import ActiveTrainerBase
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
import numpy as np


class KMeansSegmentsBalancedTrainer(ActiveTrainerBase):

    def select_new_samples(self):
        self.num_classes = self.model.num_classes

        # prediction
        features_vec, predictions_vec = self.collect_features('train')

        labeled_features_vec      = features_vec[self.dataset.train_dataset.pool]
        unlabeled_features_vec    = features_vec[self.dataset.train_dataset.available_samples]
        unlabeled_predictions_vec = predictions_vec[self.dataset.train_dataset.available_samples]

        labeled_vec_dict   = dict(zip(range(labeled_features_vec.shape[0])  , self.dataset.train_dataset.pool))
        unlabeled_vec_dict = dict(zip(range(unlabeled_features_vec.shape[0]), self.dataset.train_dataset.available_samples))

        self.log.info('building kNN space only for the labeled train features')
        # getting the labels of the train data
        labels = self.dataset.train_dataset.labels[self.dataset.train_dataset.pool]
        nbrs = KNeighborsClassifier(n_neighbors=20, weights='distance', p=1)
        nbrs.fit(labeled_features_vec, labels)

        self.log.info('performing K-Means for the unlabeled train features. K=100')
        KM = KMeans(n_clusters=100,
                    n_init=10,
                    random_state=self.rand_gen,
                    n_jobs=10)
        KM.fit(unlabeled_features_vec)

        self.log.info('for each center, find new uncertained samples')
        segments = KM.predict(unlabeled_features_vec)
        counts = np.bincount(segments, minlength=100)
        clusters_dict = dict(zip(range(100), counts))  # debug
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
        self.log.info(
            'budget_dict after updating is: {}\n the sum is {}'.format(budget_dict, budget_sum_post))  # debug
        if budget_sum_post != 1000:
            err_str = 'sum(budget_dict) equals {} instead of 1000'.format(budget_sum_post)
            self.log.error(err_str)
            raise AssertionError(err_str)

        new_indices = []
        for segment_id in range(100):
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

            dnn_predictions_for_segment = unlabeled_predictions_vec[segment_indices]
            knn_predictions_for_segment = nbrs.predict_proba(features_for_segment)
            u_vec = self.uncertainty_score(dnn_predictions_for_segment, knn_predictions_for_segment)

            self.log.info('Finding the {} most uncertained samples from segment_id={}'.format(budget_dict[segment_id], segment_id))
            if budget_dict[segment_id] > 0:
                most_uncertained_segment_indices = u_vec.argsort()[-budget_dict[segment_id]:]
                most_uncertained_indices = segment_indices[most_uncertained_segment_indices]
                new_indices_tmp =  [unlabeled_vec_dict.values()[i] for i in most_uncertained_indices]
                new_indices += new_indices_tmp
            else:
                self.log.info('for segment_id={} no new samples are chosen because budget_dict[segment_id] = 0'.format(segment_id))

        return new_indices


    def uncertainty_score(self, y_pred_dnn, y_pred_knn):
        """
        Calculates the uncertainty score based on the hamming loss of arr1 and arr2
        :param y_pred_knn: np.float32 array of the KNN probability
        :param y_pred_dnn: np.float32 array of all the predictions of the network
        :return: uncertainty score for every vector
        """
        if y_pred_knn.shape != y_pred_dnn.shape:
            err_str = 'y_pred_knn.shape != y_pred_dnn.shape ({}!={})'.format(y_pred_knn.shape, y_pred_dnn.shape)
            self.log.error(err_str)
            raise AssertionError(err_str)

        score = np.empty(shape=y_pred_knn.shape[0], dtype=np.float32)
        cls_mat = np.eye(self.num_classes, dtype=np.int32)

        for row in xrange(y_pred_knn.shape[0]):
            loss = 0.0
            y_pred = y_pred_dnn[row].reshape((1,-1))
            for cls in xrange(self.num_classes):
                cls_weight = y_pred_knn[row, cls]
                cls_true   = cls_mat[cls].reshape([1,-1])
                loss      += cls_weight * log_loss(cls_true, y_pred)
            score[row] = loss

        return score
