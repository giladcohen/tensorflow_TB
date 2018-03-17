from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from lib.testers.knn_classifier_tester import KNNClassifierTester
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from utils.misc import collect_features, calc_mutual_agreement

class MultiLayerKNNClassifierTester(KNNClassifierTester):

    def __init__(self, *args, **kwargs):
        super(MultiLayerKNNClassifierTester, self).__init__(*args, **kwargs)
        self.collected_layers = self.prm.test.test_control.COLLECTED_LAYERS
        self.apply_relu       = self.prm.test.test_control.APPLY_RELU
        self.apply_gap        = self.prm.test.test_control.APPLY_GAP

        self.relu_leakiness   = self.prm.network.system.RELU_LEAKINESS

    def test(self):
        self.log.info('Start testing {}'.format(str(self)))
        for layer in self.collected_layers:
            self.collect_layer(layer)
        self.log.info('Tester {} is done'.format(str(self)))

    def collect_layer(self, layer):
        self.log.info('Start collecting samples for layer {}'.format(layer))
        (X_train_features, y_train) = \
            collect_features(
                agent=self,
                dataset_name='train_eval',
                fetches=[self.model.net[layer], self.model.labels],
                feed_dict={self.model.dropout_keep_prob: 1.0})

        (X_test_features, y_test, test_dnn_predictions_prob) = \
            collect_features(
                agent=self,
                dataset_name='test',
                fetches=[self.model.net[layer], self.model.labels, self.model.predictions_prob],
                feed_dict={self.model.dropout_keep_prob: 1.0})

        if self.apply_relu:
            X_train_features = self.relu(X_train_features)
            X_test_features  = self.relu(X_test_features)

        if self.apply_gap:
            X_train_features = self.global_average_pool(X_train_features)
            X_test_features  = self.global_average_pool(X_test_features)
        else:
            # flatten
            X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
            X_test_features  = X_test_features.reshape(X_test_features.shape[0], -1)

        self.log.info('Fitting KNN model for layer {}...'.format(layer))
        self.knn.fit(X_train_features, y_train)

        self.log.info('Predicting test set labels from layer {} with KNN model...'.format(layer))
        test_knn_predictions_prob = self.knn.predict_proba(X_test_features)
        y_pred_knn = test_knn_predictions_prob.argmax(axis=1)
        y_pred_dnn = test_dnn_predictions_prob.argmax(axis=1)
        accuracy = np.sum(y_pred_knn == y_test) / self.dataset.test_set_size
        ma_score, md_score = calc_mutual_agreement(y_pred_dnn, y_pred_knn, y_test)

        score_str = 'score_metrics/{}/K={}/RELU={}/GAP={}'\
            .format(layer, self.knn_neighbors, self.apply_relu, self.apply_gap)
        self.tb_logger_test.log_scalar(score_str + '/accuracy', accuracy, self.global_step)
        self.tb_logger_test.log_scalar(score_str + '/ma_score', ma_score, self.global_step)
        self.tb_logger_test.log_scalar(score_str + '/md_score', md_score, self.global_step)

        # debug for screen log
        print_str = 'layer {}: accuracy: {}, ma_score={}, md_score={}'.format(layer, accuracy, ma_score, md_score)
        self.log.info(print_str)
        print(print_str)
        self.summary_writer_test.flush()

    def global_average_pool(self, x):
        if x.ndim != 4:
            err_str = 'global_average_pool must get a vector with ndim=4'
            self.log.error(err_str)
            raise AssertionError(err_str)
        return np.mean(x, axis=(1, 2))

    def relu(self, x):
        return np.where(np.less(x, 0.0), self.relu_leakiness * x, x)

    def print_stats(self):
        '''print basic test parameters'''
        super(MultiLayerKNNClassifierTester, self).print_stats()
        self.log.info(' COLLECTED_LAYERS: {}'.format(self.collected_layers))
        self.log.info(' APPLY_RELU: {}'.format(self.apply_relu))
        self.log.info(' APPLY_GAP: {}'.format(self.apply_gap))

