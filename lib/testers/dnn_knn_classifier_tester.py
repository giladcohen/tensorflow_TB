from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from lib.testers.tester_base import TesterBase
from utils.tensorboard_logging import TBLogger
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import normalized_mutual_info_score
from utils.misc import collect_features


class DNNKNNClassifierTester(TesterBase):

    def __init__(self, *args, **kwargs):
        super(DNNKNNClassifierTester, self).__init__(*args, **kwargs)

        self.pca_reduction         = self.prm.train.train_control.PCA_REDUCTION
        self.pca_embedding_dims    = self.prm.train.train_control.PCA_EMBEDDING_DIMS

        # testing parameters
        self.knn_neighbors   = self.prm.test.test_control.KNN_NEIGHBORS
        self.knn_p_norm      = self.prm.test.test_control.KNN_P_NORM
        self.knn_jobs        = self.prm.test.test_control.KNN_JOBS

        self.pca = PCA(n_components=self.pca_embedding_dims, random_state=self.rand_gen)
        self.knn = KNeighborsClassifier(n_neighbors=self.knn_neighbors, p=self.knn_p_norm, n_jobs=self.knn_jobs)

    def test(self):
        train_size = self.dataset.train_set_size
        test_size  = self.dataset.test_set_size

        self.log.info('Collecting {} train set embedding features'.format(train_size))
        (X_train_features, y_train) = \
            collect_features(
                agent=self,
                dataset_type='train_eval',
                fetches=[self.model.net['embedding_layer'], self.model.labels],
                feed_dict={self.model.dropout_keep_prob: 1.0})

        self.log.info('Collecting {} test set embedding features and DNN predictions'.format(test_size))
        (X_test_features, y_test, test_dnn_predictions_prob, test_dnn_pred) = \
            collect_features(
                agent=self,
                dataset_type='test',
                fetches=[self.model.net['embedding_layer'], self.model.labels, self.model.predictions_prob, self.model.predictions],
                feed_dict={self.model.dropout_keep_prob: 1.0})

        if self.pca_reduction:
            self.log.info('Reducing features_vec from {} dims to {} dims using PCA'.format(self.model.embedding_dims, self.pca_embedding_dims))
            X_train_features_post = self.pca.fit_transform(X_train_features)
            X_test_features_post  = self.pca.transform(X_test_features)
        else:
            X_train_features_post = X_train_features
            X_test_features_post  = X_test_features

        self.log.info('Fitting KNN model...')
        self.knn.fit(X_train_features_post, y_train)

        self.log.info('Predicting test set labels from KNN model...')
        test_knn_pred = self.knn.predict(X_test_features_post)

        # calculating different metrics
        dnn_accuracy  = np.sum(test_dnn_pred==y_test)/test_size
        knn_accuracy  = np.sum(test_knn_pred==y_test)/test_size
        dnn_nmi_score = normalized_mutual_info_score(labels_true=y_test, labels_pred=test_dnn_pred)
        knn_nmi_score = normalized_mutual_info_score(labels_true=y_test, labels_pred=test_knn_pred)

        # writing summaries
        self.tb_logger_test.log_scalar('score/dnn_accuracy', dnn_accuracy , self.global_step)
        self.tb_logger_test.log_scalar('score/dnn_NMI'     , dnn_nmi_score, self.global_step)
        self.tb_logger_test.log_scalar('score/knn_accuracy', knn_accuracy , self.global_step)
        self.tb_logger_test.log_scalar('score/knn_NMI'     , knn_nmi_score, self.global_step)

        self.summary_writer_test.flush()
        self.log.info('TEST : dnn accuracy: {}, dnn NMI score: {}\n\t   knn accuracy: {} knn NMI score: {}'
                      .format(dnn_accuracy, dnn_nmi_score, knn_accuracy, knn_nmi_score))

        if self.dump_net:
            train_features_file            = os.path.join(self.test_dir, 'train_features.npy')
            test_features_file             = os.path.join(self.test_dir, 'test_features.npy')
            test_dnn_predictions_prob_file = os.path.join(self.test_dir, 'test_dnn_predictions_prob.npy')
            test_dnn_predictions_file      = os.path.join(self.test_dir, 'test_dnn_predictions.npy')
            test_knn_predictions_file      = os.path.join(self.test_dir, 'test_knn_predictions.npy')
            train_labels_file              = os.path.join(self.test_dir, 'train_labels.npy')
            test_labels_file               = os.path.join(self.test_dir, 'test_labels.npy')

            self.log.info('Dumping train features into disk:\n{}\n{}\n{}\n{}\n{}\n{}\n{})'
                          .format(train_features_file, test_features_file, test_dnn_predictions_prob_file,
                                  test_dnn_predictions_file, test_knn_predictions_file, train_labels_file, test_labels_file))
            np.save(train_features_file           , X_train_features)
            np.save(test_features_file            , X_test_features)
            np.save(test_dnn_predictions_prob_file, test_dnn_predictions_prob)
            np.save(test_dnn_predictions_file     , test_dnn_pred)
            np.save(test_knn_predictions_file     , test_knn_pred)
            np.save(train_labels_file             , y_train)
            np.save(test_labels_file              , y_test)

        self.log.info('Tester {} is done'.format(str(self)))

    def print_stats(self):
        '''print basic test parameters'''
        super(DNNKNNClassifierTester, self).print_stats()
        self.log.info(' PCA_REDUCTION: {}'.format(self.pca_reduction))
        self.log.info(' PCA_EMBEDDING_DIMS: {}'.format(self.pca_embedding_dims))
        self.log.info(' KNN_NEIGHBORS: {}'.format(self.knn_neighbors))
        self.log.info(' KNN_P_NORM: {}'.format(self.knn_p_norm))
        self.log.info(' KNN_JOBS: {}'.format(self.knn_jobs))


