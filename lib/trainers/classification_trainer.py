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

    def __init__(self, *args, **kwargs):
        super(ClassificationTrainer, self).__init__(*args, **kwargs)
        self.eval_batch_count     = int(ceil(self.dataset.validation_dataset.size / self.eval_batch_size))
        self.last_eval_batch_size =          self.dataset.validation_dataset.size % self.eval_batch_size

    def train_step(self):
        '''Implementing one training step'''
        images, labels = self.dataset.get_mini_batch_train()
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
                dataset_type='validation',
                fetches=[self.model.labels, self.model.net['logits'], self.model.cost],
                feed_dict={self.model.dropout_keep_prob: 1.0}
            )
        score = np.average(labels  == predictions)

        # sample loss/summaries for only the first batch
        images, labels = self.dataset.get_mini_batch_validate(indices=range(self.eval_batch_size))
        (summaries, loss) = self.sess.run([self.model.summaries, self.model.cost],
                                          feed_dict={self.model.images           : images,
                                                     self.model.labels           : labels,
                                                     self.model.is_training      : False,
                                                     self.model.dropout_keep_prob: 1.0})

        self.validation_retention.add_score(score, self.global_step)
        self.tb_logger_eval.log_scalar('score', score, self.global_step)
        self.tb_logger_eval.log_scalar('best score', self.validation_retention.get_best_score(), self.global_step)
        self.summary_writer_eval.add_summary(summaries, self.global_step)
        self.summary_writer_eval.flush()
        self.log.info('EVALUATION (step={}): loss: {}, score: {}, best score: {}' \
                      .format(self.global_step, loss, score, self.validation_retention.get_best_score()))

    def print_stats(self):
        super(ClassificationTrainer, self).print_stats()
        self.log.info(' EVAL_BATCH_COUNT: {}'.format(self.eval_batch_count))
        self.log.info(' LAST_EVAL_BATCH_SIZE: {}'.format(self.last_eval_batch_size))

    def get_train_summaries(self):
        super(ClassificationTrainer, self).get_train_summaries()
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.image('input_images', self.model.images))
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.scalar('dropout_keep_prob', self.model.dropout_keep_prob))
