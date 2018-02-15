from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from lib.trainers.dynamic_model_trainer import DynamicModelTrainer


class MostUncertainedKnnDynamicTrainer(DynamicModelTrainer):

    def select_new_samples(self):

        # analyzing (prediction)
        features_vec, _ = self.collect_features('train')

        labeled_features_vec      = features_vec[self.dataset.train_dataset.pool]
        unlabeled_features_vec    = features_vec[self.dataset.train_dataset.available_samples]

        unlabeled_vec_dict = dict(zip(range(unlabeled_features_vec.shape[0]), self.dataset.train_dataset.available_samples))

        # getting the labels of the train data
        labels = self.dataset.train_dataset.labels[self.dataset.train_dataset.pool]
        n_neighbors = int(0.02 * self.dataset.train_dataset.pool_size())
        self.log.info('building kNN space only for the labeled train features. k={}'.format(n_neighbors))
        nbrs = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', p=1)
        nbrs.fit(labeled_features_vec, labels)

        # prediction
        self.log.info('Calculating the estimated labels probability based on KNN')
        estimated_labels_vec = nbrs.predict_proba(unlabeled_features_vec)
        u_vec = self.uncertainty_score(estimated_labels_vec)

        unlabeled_predictions_indices = u_vec.argsort()[-self.dataset.train_dataset.clusters:]
        new_indices = [unlabeled_vec_dict.values()[i] for i in unlabeled_predictions_indices]

        return new_indices

    def uncertainty_score(self, y_pred_knn):
        """
        Calculates the uncertainty score based on the hamming loss of arr1 and arr2
        :param y_pred_knn: np.float32 array of the KNN probability
        :return: uncertainty score for every vector
        """
        return np.power(y_pred_knn.max(axis=1), -1)
