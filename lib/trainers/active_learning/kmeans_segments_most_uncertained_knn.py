from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer_base import ActiveTrainerBase
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
import numpy as np


class KMeansSegmentsMostUncertainedKNNTrainer(ActiveTrainerBase):

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

        self.log.info('for each center, find 10 new most uncertained samples')
        segments = KM.predict(unlabeled_features_vec)
        unique, counts = np.unique(segments, return_counts=True)  # debug
        clusters_dict = dict(zip(unique, counts))                 # debug
        assert len(unique) == 100, 'the length of unique must equal 100 (number of segments'

        new_indices = []
        for segment_id in range(100):
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

            # dnn_predictions_for_segment = unlabeled_predictions_vec[segment_indices]
            knn_predictions_for_segment = nbrs.predict_proba(features_for_segment)
            u_vec = self.uncertainty_score(knn_predictions_for_segment)

            self.log.info('Finding the 10 most uncertained samples from segment_id={}'.format(segment_id))
            most_uncertained_segment_indices = u_vec.argsort()[-10:]
            most_uncertained_indices = segment_indices[most_uncertained_segment_indices]
            new_indices_tmp =  [unlabeled_vec_dict.values()[i] for i in most_uncertained_indices]
            new_indices += new_indices_tmp

        return new_indices


    def uncertainty_score(self, y_pred_knn):
        """
        Calculates the uncertainty score based on the hamming loss of arr1 and arr2
        :param y_pred_knn: np.float32 array of the KNN probability
        :return: uncertainty score for every vector
        """
        return np.power(y_pred_knn.max(axis=1), -1)
