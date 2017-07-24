from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.trainers.classification_trainer import ClassificationTrainer
from math import ceil
import numpy as np


class PassiveTrainer(ClassificationTrainer):
    """Implementing passive trainer
    Using the entire labeled trainset for training"""

    def __init__(self, *args, **kwargs):
        super(PassiveTrainer, self).__init__(*args, **kwargs)
        self.eval_batch_count     = int(ceil(self.dataset.validation_set.size / self.eval_batch_size))
        self.last_eval_batch_size = self.dataset.validation_set.size % self.eval_batch_size

    def __str__(self):
        return 'PassiveTrainer'

    def train_step(self):
        '''Implementing one training step'''
        _, _, images_aug, labels_aug = self.dataset.get_mini_batch_train()
        self.sess.run(self.model.train_op, feed_dict={self.model.images: images_aug,
                                                      self.model.labels: labels_aug,
                                                      self.model.is_training: True})

    def eval_step(self):
        '''Implementing one evaluation step.'''
        self._logger.info('start running eval within training. global_step={}'.format(self.global_step))
        total_prediction, correct_prediction = 0, 0
        for i in range(self.eval_batch_count):
            b = i * self.eval_batch_size
            if i < (self.eval_batch_count - 1) or (self.last_eval_batch_size == 0):
                e = (i + 1) * self.eval_batch_size
            else:
                e = i * self.eval_batch_size + self.last_eval_batch_size
            images, labels, _, _ = self.dataset.get_mini_batch_eval(indices=range(b, e))
            (summaries, loss, predictions, train_step) = self.sess.run(
                [self.model.summaries, self.model.cost,
                 self.model.predictions, self.model.global_step],
                feed_dict={self.model.images     : images,
                           self.model.labels     : labels,
                           self.model.is_training: False})

            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(labels == predictions)
            total_prediction += predictions.shape[0]
        if total_prediction != self.dataset.validation_set.size:
            self._logger.error('total_prediction equals {} instead of {}'.format(total_prediction,
                                                                                 self.dataset.validation_set.size))
        precision = correct_prediction / total_prediction
        self.best_precision = max(precision, self.best_precision)

        precision_summ = tf.Summary()
        precision_summ.value.add(tag='Precision', simple_value=precision)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(tag='Best Precision', simple_value=self.best_precision)
        self.summary_writer_eval.add_summary(precision_summ, train_step)
        self.summary_writer_eval.add_summary(best_precision_summ, train_step)
        self.summary_writer_eval.add_summary(summaries, train_step)
        self.summary_writer_eval.flush()

        self._logger.info('EVALUATION: loss: {}, precision: {}, best precision: {}'.format(loss, precision, self.best_precision))

    def print_stats(self):
        super(PassiveTrainer, self).print_stats()
        self._logger.info(' EVAL_BATCH_COUNT: {}'.format(self.eval_batch_count))
        self._logger.info(' LAST_EVAL_BATCH_SIZE: {}'.format(self.last_eval_batch_size))
