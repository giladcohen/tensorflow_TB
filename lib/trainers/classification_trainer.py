from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.trainers.trainer_base import TrainerBase
from math import ceil
from lib.base.collections import TRAIN_SUMMARIES
import numpy as np
from utils.misc import collect_features


class ClassificationTrainer(TrainerBase):
    """Implementing classification trainer
    Using the entire labeled trainset for training"""

    def train_step(self):
        '''Implementing one training step'''
        images, labels = self.dataset.get_mini_batch('train', self.plain_sess)
        _ , self.global_step = self.sess.run([self.model.train_op, self.model.global_step],
                                              feed_dict={self.model.images: images,
                                                         self.model.labels: labels,
                                                         self.model.is_training: True})

    def eval_step(self):
        '''Implementing one evaluation step.'''
        self.log.info('start running eval within training. global_step={}'.format(self.global_step))
        (labels, predictions) = \
            collect_features(
                agent=self,
                dataset_name='validation',
                fetches=[self.model.labels, self.model.predictions],
                feed_dict={self.model.dropout_keep_prob: 1.0})
        score = np.average(labels == predictions)

        # sample loss/summaries for only the first batch
        (summaries, loss) = self.sample_stats(dataset_name='validation')

        self.validation_retention.add_score(score, self.global_step)
        self.tb_logger_validation.log_scalar('score', score, self.global_step)
        self.tb_logger_validation.log_scalar('best score', self.validation_retention.get_best_score(), self.global_step)
        self.summary_writer_validation.add_summary(summaries, self.global_step)
        self.summary_writer_validation.flush()
        self.log.info('EVALUATION (step={}): loss: {}, score: {}, best score: {}' \
                      .format(self.global_step, loss, score, self.validation_retention.get_best_score()))

    def test_step(self):
        '''Implementing one test step.'''
        self.log.info('start running test within training. global_step={}'.format(self.global_step))
        (labels, predictions) = \
            collect_features(
                agent=self,
                dataset_name='test',
                fetches=[self.model.labels, self.model.predictions],
                feed_dict={self.model.dropout_keep_prob: 1.0})
        score = np.average(labels == predictions)

        # sample loss/summaries for only the first batch
        (summaries, loss) = self.sample_stats(dataset_name='test')

        self.test_retention.add_score(score, self.global_step)
        self.tb_logger_test.log_scalar('score', score, self.global_step)
        self.tb_logger_test.log_scalar('best score', self.test_retention.get_best_score(), self.global_step)
        self.summary_writer_test.add_summary(summaries, self.global_step)
        self.summary_writer_test.flush()
        self.log.info('TEST (step={}): loss: {}, score: {}, best score: {}' \
                      .format(self.global_step, loss, score, self.test_retention.get_best_score()))

    def sample_stats(self, dataset_name):
        """Sampling validation/test summary and loss only for one eval batch."""
        if dataset_name == 'validation':
            self.plain_sess.run(self.dataset.validation_iterator.initializer)
        elif dataset_name == 'test':
            self.plain_sess.run(self.dataset.test_iterator.initializer)
        else:
            err_str = 'sample_stats must be called only with dataset=validation/test, not {}'.format(dataset_name)
            self.log.error(err_str)
            raise AssertionError(err_str)

        images, labels = self.dataset.get_mini_batch(dataset_name, self.plain_sess)
        (summaries, loss) = self.plain_sess.run([self.model.summaries, self.model.cost],
                                          feed_dict={self.model.images: images,
                                                     self.model.labels: labels,
                                                     self.model.is_training: False,
                                                     self.model.dropout_keep_prob: 1.0})
        return summaries, loss

    def get_train_summaries(self):
        super(ClassificationTrainer, self).get_train_summaries()
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.image('input_images', self.model.images))
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.scalar('dropout_keep_prob', self.model.dropout_keep_prob))
