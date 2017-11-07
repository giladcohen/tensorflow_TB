from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer_base import ActiveTrainerBase
import numpy as np
# from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn import metrics


class CrossEntropyTrainer(ActiveTrainerBase):

    def __init__(self, *args, **kwargs):
        super(CrossEntropyTrainer, self).__init__(*args, **kwargs)
        self.num_classes = self.model.num_classes

    def select_new_samples(self):

        # analyzing (evaluation)
        features_vec, predictions_vec = self.collect_features('train')

        labeled_features_vec      = features_vec[self.dataset.train_dataset.pool]
        unlabeled_features_vec    = features_vec[self.dataset.train_dataset.available_samples]
        unlabeled_predictions_vec = predictions_vec[self.dataset.train_dataset.available_samples]

        labeled_vec_dict   = dict(zip(range(labeled_features_vec.shape[0])  , self.dataset.train_dataset.pool))
        unlabeled_vec_dict = dict(zip(range(unlabeled_features_vec.shape[0]), self.dataset.train_dataset.available_samples))

        self.log.info('building kNN space only for the labeled train features')
        # getting the labels of the train data
        labels = self.dataset.train_dataset.labels[self.dataset.train_dataset.pool]
        nbrs = KNeighborsClassifier(n_neighbors=20, weights='distance', p=1)  # TODO(gilad): should be increased for higher number of new samples
        nbrs.fit(labeled_features_vec, labels)

        # prediction
        self.log.info('Calculating the estimated labels probability based on KNN')
        estimated_labels_vec = nbrs.predict_proba(unlabeled_features_vec)
        u_vec = self.uncertainty_score(estimated_labels_vec, unlabeled_predictions_vec)

        unlabeled_predictions_indices = u_vec.argsort()[-self.dataset.train_dataset.clusters:]
        new_indices = [unlabeled_vec_dict.values()[i] for i in unlabeled_predictions_indices]

        return new_indices

    def uncertainty_score(self, y_pred_knn, y_pred_dnn):
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
