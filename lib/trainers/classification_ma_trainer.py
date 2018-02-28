from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.classification_trainer import ClassificationTrainer
import numpy as np
from utils.misc import collect_features, calc_mutual_agreement
from sklearn.neighbors import KNeighborsClassifier

class ClassificationMATrainer(ClassificationTrainer):
    """Implementing classification trainer
    Using the entire labeled trainset for training"""

    def __init__(self, *args, **kwargs):
        super(ClassificationMATrainer, self).__init__(*args, **kwargs)
        self.dnn_train_handle = 'train'       # either train      or train_random
        self.knn_train_handle = 'train_eval'  # either train_eval or train_random_eval

        # testing parameters
        self.knn_neighbors   = self.prm.test.test_control.KNN_NEIGHBORS
        self.knn_norm        = self.prm.test.test_control.KNN_NORM
        self.knn_weights     = self.prm.test.test_control.KNN_WEIGHTS
        self.knn_jobs        = self.prm.test.test_control.KNN_JOBS

        if self.knn_norm not in ['L1', 'L2']:
            err_str = 'knn_norm {} is not supported'.format(self.knn_norm)
            self.log.error(err_str)
            raise AssertionError(err_str)

        self.knn = KNeighborsClassifier(
            n_neighbors=self.knn_neighbors,
            weights=self.knn_weights,
            p=int(self.knn_norm[-1]),
            n_jobs=self.knn_jobs)

    def train_step(self):
        '''Implementing one training step'''
        _ , images, labels = self.dataset.get_mini_batch(self.dnn_train_handle, self.plain_sess)
        _ , self.global_step = self.sess.run([self.model.train_op, self.model.global_step],
                                              feed_dict={self.model.images: images,
                                                         self.model.labels: labels,
                                                         self.model.is_training: True})

    def test_step(self):
        '''Implementing one test step.'''
        self.log.info('start running test within training. global_step={}'.format(self.global_step))
        self.log.info('Collecting {} {} set embedding features'.format(self.dataset.train_set_size, self.knn_train_handle))
        (X_train_features, y_train) = \
            collect_features(
                agent=self,
                dataset_name=self.knn_train_handle,
                fetches=[self.model.net['embedding_layer'], self.model.labels],
                feed_dict={self.model.dropout_keep_prob: 1.0})

        self.log.info('Collecting {} test set embedding features and DNN predictions'.format(self.dataset.test_set_size))
        (X_test_features, y_test, test_dnn_predictions_prob) = \
            collect_features(
                agent=self,
                dataset_name='test',
                fetches=[self.model.net['embedding_layer'], self.model.labels, self.model.predictions_prob],
                feed_dict={self.model.dropout_keep_prob: 1.0})

        self.log.info('Fitting KNN model...')
        self.knn.fit(X_train_features, y_train)

        y_pred_dnn = test_dnn_predictions_prob.argmax(axis=1)
        self.log.info('Predicting test set labels from KNN model...')
        test_knn_predictions_prob = self.knn.predict_proba(X_test_features)
        y_pred_knn = test_knn_predictions_prob.argmax(axis=1)

        ma_score, md_score = calc_mutual_agreement(y_pred_dnn, y_pred_knn, y_test)
        dnn_score = np.average(y_test == y_pred_dnn)
        knn_score = np.average(y_test == y_pred_knn)

        # sample loss/summaries for only the first batch
        (summaries, loss) = self.sample_stats(dataset_name='test')

        self.test_retention.add_score(dnn_score, self.global_step)

        self.tb_logger_test.log_scalar('ma_score', ma_score, self.global_step)
        self.tb_logger_test.log_scalar('md_score', md_score, self.global_step)
        self.tb_logger_test.log_scalar('score', dnn_score, self.global_step)
        self.tb_logger_test.log_scalar('best score', self.test_retention.get_best_score(), self.global_step)
        self.tb_logger_test.log_scalar('knn_score', knn_score, self.global_step)
        self.summary_writer_test.add_summary(summaries, self.global_step)
        self.summary_writer_test.flush()
        self.log.info('TEST (step={}): loss: {}, dnn_score: {}, knn_score: {}, best score: {}' \
                      .format(self.global_step, loss, dnn_score, knn_score, self.test_retention.get_best_score()))
