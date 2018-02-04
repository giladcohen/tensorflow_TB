from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.testers.knn_classifier_tester import KNNClassifierTester
import os
from sklearn.neighbors import KNeighborsClassifier
from utils.misc import collect_features

class EnsembleTester(KNNClassifierTester):

    def __init__(self, *args, **kwargs):
        super(EnsembleTester, self).__init__(*args, **kwargs)
        self.log_dir_list          = self.prm.test.ensemble.LOG_DIR_LIST

        # variables
        self.ensemble_size         = len(self.log_dir_list)
        self.checkpoint_file_list  = self.get_checkpoint_file_list()
        self.test_dir_list         = self.get_test_dir_list()

    def get_checkpoint_file_list(self):
        """Getting a list containing all the checkpoint files in the ensemble"""
        checkpoint_file_list = []
        for i in xrange(self.ensemble_size):
            dir_basename = os.path.basename(self.checkpoint_dir)
            checkpoint_file_list.append(os.path.join(self.log_dir_list[i], dir_basename, self.checkpoint_file))
        return checkpoint_file_list

    def get_test_dir_list(self):
        """Getting a list containing all the test dirs in the ensemble"""
        test_dir_list = []
        for i in xrange(self.ensemble_size):
            dir_basename = os.path.basename(self.test_dir)
            test_dir_list.append(os.path.join(self.log_dir_list[i], dir_basename))
        return test_dir_list

    def finalize_graph(self):
        self.dataset.set_handles(self.plain_sess)

    def set_params(self):
        self.log.info('Not setting params for ensemble tester')

    def test(self):
        """Testing ensemble"""
        X_train_features, \
        y_train, \
        X_test_features, \
        y_test, \
        test_dnn_predictions_prob = self.load_features()

        if self.decision_method == 'dnn_median':
            y_median = np.median(test_dnn_predictions_prob, axis=1)   # median over all ensembles.
                                                                      # shape=[self.test_set_size, self.num_classes]
            y_pred = y_median.argmax(axis=1).astype(np.int32)
        elif self.decision_method == 'dnn_average':
            y_average = np.average(test_dnn_predictions_prob, axis=1) # median over all ensembles.
                                                                      # shape = [self.test_set_size, self.num_classes]
            y_pred = y_average.argmax(axis=1).astype(np.int32)
        elif self.decision_method == 'knn_aggregate_nc_dropout':
            test_knn_predictions_prob_ensemble_mat = np.empty(shape=[self.dataset.test_set_size, self.ensemble_size, self.num_classes], dtype=np.float32)
            knn_models = []
            for i in xrange(self.ensemble_size):
                self.log.info('Constructing KNN model for net #{}'.format(i))
                knn_models.append(KNeighborsClassifier(
                    n_neighbors=self.knn_neighbors,
                    weights=self.knn_weights,
                    p=int(self.knn_norm[-1]),
                    n_jobs=self.knn_jobs))

            number_of_predictions = 20
            for i in xrange(self.ensemble_size):
                test_knn_predictions_prob_sum = np.zeros(shape=[self.dataset.test_set_size, self.num_classes], dtype=np.float32)
                self.log.info('Training KNN model for net #{}'.format(i))
                X_train_features[:, i, :] = self.apply_pca(X_train_features[:, i, :], fit=True)
                knn_models[i].fit(X_train_features[:, i, :], y_train[:, 0])
                self.log.info('loading KNN model parameters for net #{}'.format(i))
                self.saver.restore(self.plain_sess, self.checkpoint_file_list[i])
                self.log.info('Predicting KNN model for net #{} using NC dropout'.format(i))
                for k in xrange(number_of_predictions):
                    (X_test_features, ) = \
                        collect_features(
                            agent=self,
                            dataset_type='test',
                            fetches=[self.model.net['embedding_layer']],
                            feed_dict={self.model.dropout_keep_prob: 0.5})
                    X_test_features = self.apply_pca(X_test_features, fit=False)
                    test_knn_predictions_prob_sum += knn_models[i].predict_proba(X_test_features)
                test_knn_predictions_prob_ensemble_mat[:, i, :] = test_knn_predictions_prob_sum
            test_knn_predictions_prob_mat = np.average(test_knn_predictions_prob_ensemble_mat, axis=1)  # shape=[self.dataset.test_set_size, self.num_classes]
            y_pred = test_knn_predictions_prob_mat.argmax(axis=1)

        accuracy = np.sum(y_pred == y_test[:, 0]) / self.dataset.test_set_size
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
        X_train_features          = np.empty(shape=[self.dataset.train_set_size, self.ensemble_size, self.model.embedding_dims], dtype=np.float32)
        y_train                   = np.empty(shape=[self.dataset.train_set_size, self.ensemble_size], dtype=np.int32)
        X_test_features           = np.empty(shape=[self.dataset.test_set_size , self.ensemble_size, self.model.embedding_dims], dtype=np.float32)
        y_test                    = np.empty(shape=[self.dataset.test_set_size , self.ensemble_size], dtype=np.int32)
        test_dnn_predictions_prob = np.empty(shape=[self.dataset.test_set_size , self.ensemble_size, self.num_classes], dtype=np.float32)

        self.log.info("Start loading entire ensemble features from disk")
        for i in xrange(self.ensemble_size):
            X_train_features_i, \
            X_test_features_i, \
            test_dnn_predictions_prob_i, \
            y_train_i, \
            y_test_i = self.fetch_dump_data_features(self.test_dir_list[i])

            X_train_features[:, i, :]          = X_train_features_i
            y_train[:, i]                      = y_train_i
            X_test_features[:, i, :]           = X_test_features_i
            y_test[:, i]                       = y_test_i
            test_dnn_predictions_prob[:, i, :] = test_dnn_predictions_prob_i

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
