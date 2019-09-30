from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_TB.lib.trainers.active_trainer import ActiveTrainer
from tensorflow_TB.utils.misc import collect_features, calc_mutual_agreement
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class RandomSamplerTrainerQAD(ActiveTrainer):

    def __init__(self, *args, **kwargs):
        super(RandomSamplerTrainerQAD, self).__init__(*args, **kwargs)
        self.knn = KNeighborsClassifier(
            n_neighbors=30,
            weights='uniform',
            p=1,
            n_jobs=20)

    def select_new_samples(self):
        return None  # will result in random selection of samples

    # def eval_step(self):
    #     '''Implementing one evaluation step.'''
    #     self.log.info('start running eval within training. global_step={}'.format(self.global_step))
    #     # collecting train features and train labels
    #     (y_train, X_train_features) = \
    #         collect_features(
    #             agent=self,
    #             dataset_name='train_pool_eval',
    #             fetches=[self.model.labels, self.model.net['embedding_layer']],
    #             feed_dict={self.model.dropout_keep_prob: 1.0})
    #     self.log.info('Fitting KNN model...')
    #     self.knn.fit(X_train_features, y_train)
    #
    #     (y_test, y_pred, X_test_features) = \
    #         collect_features(
    #             agent=self,
    #             dataset_name='validation',
    #             fetches=[self.model.labels, self.model.predictions, self.model.net['embedding_layer']],
    #             feed_dict={self.model.dropout_keep_prob: 1.0})
    #     score = np.average(y_test == y_pred)
    #     test_knn_predictions_prob = self.knn.predict_proba(X_test_features)
    #     y_pred_knn = test_knn_predictions_prob.argmax(axis=1)
    #     ma_score, md_score = calc_mutual_agreement(y_pred, y_pred_knn, y_test)
    #
    #     # sample loss/summaries for only the first batch
    #     (summaries, loss) = self.sample_stats(dataset_name='validation')
    #
    #     self.validation_retention.add_score(score, self.global_step)
    #     self.tb_logger_validation.log_scalar('score', score, self.global_step)
    #     self.tb_logger_validation.log_scalar('best score', self.validation_retention.get_best_score(), self.global_step)
    #     self.tb_logger_validation.log_scalar('ma_score', ma_score, self.global_step)
    #     self.tb_logger_validation.log_scalar('md_score', md_score, self.global_step)
    #     self.summary_writer_validation.add_summary(summaries, self.global_step)
    #     self.summary_writer_validation.flush()
    #     self.log.info('EVALUATION (step={}): loss: {}, score: {}, best score: {}, ma_score: {}, md_score: {}' \
    #                   .format(self.global_step, loss, score, self.validation_retention.get_best_score(), ma_score, md_score))

    def test_step(self):
        '''Implementing one test step.'''
        self.log.info('start running test within training. global_step={}'.format(self.global_step))
        (y_train, X_train_features) = \
            collect_features(
                agent=self,
                dataset_name='train_pool_eval',
                fetches=[self.model.labels, self.model.net['embedding_layer']],
                feed_dict={self.model.dropout_keep_prob: 1.0})
        self.log.info('Fitting KNN model...')
        self.knn.fit(X_train_features, y_train)

        (y_test, y_pred, X_test_features) = \
            collect_features(
                agent=self,
                dataset_name='test',
                fetches=[self.model.labels, self.model.predictions, self.model.net['embedding_layer']],
                feed_dict={self.model.dropout_keep_prob: 1.0})
        score = np.average(y_test == y_pred)
        test_knn_predictions_prob = self.knn.predict_proba(X_test_features)
        y_pred_knn = test_knn_predictions_prob.argmax(axis=1)
        ma_score, md_score = calc_mutual_agreement(y_pred, y_pred_knn, y_test)

        # sample loss/summaries for only the first batch
        (summaries, loss) = self.sample_stats(dataset_name='test')

        self.test_retention.add_score(score, self.global_step)
        self.tb_logger_test.log_scalar('score', score, self.global_step)
        self.tb_logger_test.log_scalar('best score', self.test_retention.get_best_score(), self.global_step)
        self.tb_logger_test.log_scalar('ma_score', ma_score, self.global_step)
        self.tb_logger_test.log_scalar('md_score', md_score, self.global_step)
        self.summary_writer_test.add_summary(summaries, self.global_step)
        self.summary_writer_test.flush()
        self.log.info('TEST (step={}): loss: {}, score: {}, best score: {}, ma_score: {}, md_score: {}' \
                      .format(self.global_step, loss, score, self.test_retention.get_best_score(), ma_score, md_score))
