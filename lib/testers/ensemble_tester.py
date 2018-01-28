from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.testers.tester_base import TesterBase
import tensorflow as tf
from utils.tensorboard_logging import TBLogger
import os

class EnsembleTester(TesterBase):

    def __init__(self, *args, **kwargs):
        super(EnsembleTester, self).__init__(*args, **kwargs)

        self.log_dir_list          = self.prm.test.ensemble.LOG_DIR_LIST
        self.decision_method       = self.prm.test.ensemble.DECISION_METHOD

        self.train_set_size        = self.prm.dataset.TRAIN_SET_SIZE
        self.test_set_size         = self.prm.dataset.TEST_SET_SIZE

        self.pca_reduction         = self.prm.train.train_control.PCA_REDUCTION
        self.pca_embedding_dims    = self.prm.train.train_control.PCA_EMBEDDING_DIMS

        # testing parameters
        self.knn_neighbors         = self.prm.test.test_control.KNN_NEIGHBORS
        self.knn_norm              = self.prm.test.test_control.KNN_NORM
        self.knn_weights           = self.prm.test.test_control.KNN_WEIGHTS
        self.knn_jobs              = self.prm.test.test_control.KNN_JOBS

        # variables
        self.ensemble_size = len(self.log_dir_list)
        self.num_classes = int(self.dataset.dataset_name[5:])

    def build_test_env(self):
        self.log.info("Starting building the test environment")
        self.summary_writer_test = tf.summary.FileWriter(self.test_dir)
        self.tb_logger_test = TBLogger(self.summary_writer_test)

    def finalize_graph(self):
        self.dataset.set_handles(self.plain_sess)

    def set_params(self):
        self.log.info('Not setting params for ensemble tester')

    def test(self):
        """Testing ensemble"""
        # loading the entire train/test features and labels
        if self.load_from_disk:
            # useful for DNN scores
            X_train_features, y_train, X_test_features, y_test, test_dnn_predictions_prob = self.load_features()

        if self.decision_method == 'dnn_median':
            y_median = np.median(test_dnn_predictions_prob, axis=0)   # median over all ensembles.
                                                                      # shape=[self.test_set_size, self.num_classes]
            y_pred = y_median.argmax(axis=1).astype(np.int32)
        elif self.decision_method == 'dnn_average':
            y_average = np.average(test_dnn_predictions_prob, axis=0) # median over all ensembles.
                                                                      # shape = [self.test_set_size, self.num_classes]
            y_pred = y_average.argmax(axis=1).astype(np.int32)

        accuracy = np.sum(y_pred == y_test[0]) / self.test_set_size
        self.tb_logger_test.log_scalar('score_metrics/ensemble_' + self.decision_method + '_accuracy', accuracy, self.global_step)

        score_str = 'score_metrics/ensemble_' + self.decision_method + '_accuracy'
        print_str = '{}: {}'.format(score_str, accuracy)
        self.log.info(print_str)
        print(print_str)

        self.summary_writer_test.flush()

    def load_features(self):
        """Loading the train/test features from pretrained networks of an entire ensemble
        X_train_features.shape          = [ensemble_size, train_size(50000), embedding_size(640)]
        y_train.shape                   = [ensemble_size, train_size(50000)]
        X_test_features.shape           = [ensemble_size, test_size(10000), embedding_size(640)]
        test_dnn_predictions_prob.shape = [ensemble_size, test_size(10000), num_classes(10)]
        """
        X_train_features          = np.empty(shape=[self.ensemble_size, self.train_set_size, self.embedding_dims], dtype=np.float32)
        y_train                   = np.empty(shape=[self.ensemble_size, self.train_set_size], dtype=np.int32)
        X_test_features           = np.empty(shape=[self.ensemble_size, self.test_set_size, self.embedding_dims], dtype=np.float32)
        y_test                    = np.empty(shape=[self.ensemble_size, self.test_set_size], dtype=np.int32)
        test_dnn_predictions_prob = np.empty(shape=[self.ensemble_size, self.test_set_size, self.num_classes], dtype=np.float32)

        self.log.info("Start loading entire ensemble features from disk")
        for i in xrange(self.ensemble_size):
            logdir = self.log_dir_list[i]
            test_dir = os.path.join(logdir, 'test')

            train_features_file            = os.path.join(test_dir, 'train_features.npy')
            train_labels_file              = os.path.join(test_dir, 'train_labels.npy')
            test_features_file             = os.path.join(test_dir, 'test_features.npy')
            test_labels_file               = os.path.join(test_dir, 'test_labels.npy')
            test_dnn_predictions_prob_file = os.path.join(test_dir, 'test_dnn_predictions_prob.npy')

            X_train_features[i]            = np.load(train_features_file)
            y_train[i]                     = np.load(train_labels_file)
            X_test_features[i]             = np.load(test_features_file)
            y_test[i]                      = np.load(test_labels_file)
            test_dnn_predictions_prob[i]   = np.load(test_dnn_predictions_prob_file)

        # assert labels are the same for every network
        for i in xrange(self.ensemble_size):
            if not (y_train[i] == y_train[0]).all():
                err_str = 'y_train[{}] values do not match y_train[0] values'.format(i)
                self.log.error(err_str)
                raise AssertionError(err_str)

            if not (y_test[i] == y_test[0]).all():
                err_str = 'y_test[{}] values do not match y_test[0] values'.format(i)
                self.log.error(err_str)
                raise AssertionError(err_str)

        return X_train_features, y_train, X_test_features, y_test, test_dnn_predictions_prob

    def print_stats(self):
        super(EnsembleTester, self).print_stats()
        self.log.info(' LOG_DIR_LIST: {}'.format(self.log_dir_list))
        self.log.info(' DECISION_METHOD: {}'.format(self.decision_method))
        self.log.info(' PCA_REDUCTION: {}'.format(self.pca_reduction))
        self.log.info(' PCA_EMBEDDING_DIMS: {}'.format(self.pca_embedding_dims))
        self.log.info(' KNN_NEIGHBORS: {}'.format(self.knn_neighbors))
        self.log.info(' KNN_NORM: {}'.format(self.knn_norm))
        self.log.info(' KNN_WEIGHTS: {}'.format(self.knn_weights))
        self.log.info(' KNN_JOBS: {}'.format(self.knn_jobs))
