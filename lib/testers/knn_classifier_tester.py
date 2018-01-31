from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from lib.testers.tester_base import TesterBase
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import normalized_mutual_info_score
from utils.misc import collect_features


class KNNClassifierTester(TesterBase):

    def __init__(self, *args, **kwargs):
        super(KNNClassifierTester, self).__init__(*args, **kwargs)
        self.decision_method = self.prm.test.test_control.DECISION_METHOD

        self.pca_reduction         = self.prm.train.train_control.PCA_REDUCTION
        self.pca_embedding_dims    = self.prm.train.train_control.PCA_EMBEDDING_DIMS

        # testing parameters
        self.knn_neighbors   = self.prm.test.test_control.KNN_NEIGHBORS
        self.knn_norm        = self.prm.test.test_control.KNN_NORM
        self.knn_weights     = self.prm.test.test_control.KNN_WEIGHTS
        self.knn_jobs        = self.prm.test.test_control.KNN_JOBS

        self.pca = PCA(n_components=self.pca_embedding_dims, random_state=self.rand_gen)

        if self.knn_norm not in ['L1', 'L2']:
            err_str = 'knn_norm {} is not supported'.format(self.knn_norm)
            self.log.error(err_str)
            raise AssertionError(err_str)

        self.knn = KNeighborsClassifier(
            n_neighbors=self.knn_neighbors,
            weights=self.knn_weights,
            p=int(self.knn_norm[-1]),
            n_jobs=self.knn_jobs)

    def fetch_dump_data_features(self):
        """Optionally fetching precomputed train/test features, and labels."""
        train_features_file            = os.path.join(self.test_dir, 'train_features.npy')
        test_features_file             = os.path.join(self.test_dir, 'test_features.npy')
        test_dnn_predictions_prob_file = os.path.join(self.test_dir, 'test_dnn_predictions_prob.npy')
        train_labels_file              = os.path.join(self.test_dir, 'train_labels.npy')
        test_labels_file               = os.path.join(self.test_dir, 'test_labels.npy')

        if self.load_from_disk:
            self.log.info('Loading {}/{} train/test set embedding features from disk'.format(self.dataset.train_set_size, self.dataset.test_set_size))
            X_train_features          = np.load(train_features_file)
            y_train                   = np.load(train_labels_file)
            X_test_features           = np.load(test_features_file)
            y_test                    = np.load(test_labels_file)
            test_dnn_predictions_prob = np.load(test_dnn_predictions_prob_file)
        else:
            self.log.info('Collecting {} train set embedding features'.format(self.dataset.train_set_size))
            (X_train_features, y_train) = \
                collect_features(
                    agent=self,
                    dataset_type='train_eval',
                    fetches=[self.model.net['embedding_layer'], self.model.labels],
                    feed_dict={self.model.dropout_keep_prob: 1.0})

            self.log.info('Collecting {} test set embedding features and DNN predictions'.format(self.dataset.test_set_size))
            (X_test_features, y_test, test_dnn_predictions_prob) = \
                collect_features(
                    agent=self,
                    dataset_type='test',
                    fetches=[self.model.net['embedding_layer'], self.model.labels, self.model.predictions_prob],
                    feed_dict={self.model.dropout_keep_prob: 1.0})

        if self.dump_net:
            self.log.info('Dumping train features into disk:\n{}\n{}\n{}\n{}\n{}'
                          .format(train_features_file, test_features_file, test_dnn_predictions_prob_file, train_labels_file, test_labels_file))
            np.save(train_features_file           , X_train_features)
            np.save(test_features_file            , X_test_features)
            np.save(test_dnn_predictions_prob_file, test_dnn_predictions_prob)
            np.save(train_labels_file             , y_train)
            np.save(test_labels_file              , y_test)

        return X_train_features, X_test_features, test_dnn_predictions_prob, y_train, y_test

    def test(self):
        X_train_features, \
        X_test_features, \
        test_dnn_predictions_prob, \
        y_train, \
        y_test = self.fetch_dump_data_features()

        if self.pca_reduction:
            self.log.info('Reducing features_vec from {} dims to {} dims using PCA'.format(self.model.embedding_dims, self.pca_embedding_dims))
            X_train_features_post = self.pca.fit_transform(X_train_features)
            X_test_features_post  = self.pca.transform(X_test_features)
        else:
            X_train_features_post = X_train_features
            X_test_features_post  = X_test_features

        self.log.info('Fitting KNN model...')
        self.knn.fit(X_train_features_post, y_train)

        if self.decision_method == 'dnn_accuracy':
            y_pred = test_dnn_predictions_prob.argmax(axis=1)
        elif self.decision_method == 'knn_accuracy':
            self.log.info('Predicting precomputed test set labels from KNN model...')
            test_knn_predictions_prob = self.knn.predict_proba(X_test_features_post)
            y_pred = test_knn_predictions_prob.argmax(axis=1)

        accuracy = np.sum(y_pred==y_test)/self.dataset.test_set_size

        # writing summaries
        score_str = 'score_metrics/K={}/PCA={}/norm={}/weights={}/decision_method={}'\
            .format(self.knn_neighbors, self.pca_embedding_dims, self.knn_norm, self.knn_weights, self.decision_method)
        self.tb_logger_test.log_scalar(score_str, accuracy, self.global_step)
        print_str = '{}: accuracy= {}'.format(score_str, accuracy)
        self.log.info(print_str)
        print(print_str)
        self.summary_writer_test.flush()

        self.log.info('Tester {} is done'.format(str(self)))

    def print_stats(self):
        '''print basic test parameters'''
        super(KNNClassifierTester, self).print_stats()
        self.log.info(' DECISION_METHOD: {}'.format(self.decision_method))
        self.log.info(' PCA_REDUCTION: {}'.format(self.pca_reduction))
        self.log.info(' PCA_EMBEDDING_DIMS: {}'.format(self.pca_embedding_dims))
        self.log.info(' KNN_NEIGHBORS: {}'.format(self.knn_neighbors))
        self.log.info(' KNN_NORM: {}'.format(self.knn_norm))
        self.log.info(' KNN_WEIGHTS: {}'.format(self.knn_weights))
        self.log.info(' KNN_JOBS: {}'.format(self.knn_jobs))


