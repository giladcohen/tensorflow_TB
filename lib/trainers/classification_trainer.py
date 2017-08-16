from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.trainers.trainer_base import TrainerBase
from math import ceil
from lib.base.collections import TRAIN_SUMMARIES
import numpy as np


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
        total_samples, total_score = 0, 0
        for i in range(self.eval_batch_count):
            b = i * self.eval_batch_size
            if i < (self.eval_batch_count - 1) or (self.last_eval_batch_size == 0):
                e = (i + 1) * self.eval_batch_size
            else:
                e = i * self.eval_batch_size + self.last_eval_batch_size
            images, labels = self.dataset.get_mini_batch_validate(indices=range(b, e))
            (summaries, loss, train_step, predictions) = self.sess.run(
                [self.model.summaries, self.model.cost,
                 self.model.global_step, self.model.predictions],
                feed_dict={self.model.images     : images,
                           self.model.labels     : labels,
                           self.model.is_training: False})

            total_score   += np.sum(labels == predictions)
            total_samples += images.shape[0]
        if total_samples != self.dataset.validation_dataset.size:
            self.log.error('total_samples equals {} instead of {}'.format(total_samples, self.dataset.validation_set.size))
        score = total_score / total_samples
        self.retention.add_score(score, train_step)

        self.tb_logger_eval.log_scalar('score', score, train_step)
        self.tb_logger_eval.log_scalar('best score', self.retention.get_best_score(), train_step)
        self.summary_writer_eval.add_summary(summaries, train_step)
        self.summary_writer_eval.flush()
        self.log.info('EVALUATION (step={}): loss: {}, score: {}, best score: {}' \
                      .format(train_step, loss, score, self.retention.get_best_score()))

    def print_stats(self):
        super(ClassificationTrainer, self).print_stats()
        self.log.info(' EVAL_BATCH_COUNT: {}'.format(self.eval_batch_count))
        self.log.info(' LAST_EVAL_BATCH_SIZE: {}'.format(self.last_eval_batch_size))

    def get_train_summaries(self):
        super(ClassificationTrainer, self).get_train_summaries()
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.image('input_images', self.model.images))
