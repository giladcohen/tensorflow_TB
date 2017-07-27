from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import tensorflow as tf

import utils
from lib.trainers.train_base import TrainBase

class ClassificationTrainerBase(TrainBase):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super(ClassificationTrainerBase, self).__init__(*args, **kwargs)
        self.best_precision = 0.0
        self.global_step    = 0
        self._activate_eval = True
        self.eval_steps = self.prm.train.train_control.EVAL_STEPS
        self.evals_in_epoch = self.prm.train.train_control.EVALS_IN_EPOCH
        if self.eval_steps is None:
            self.log.warning('EVAL_STEPS is None. Setting EVAL_STEPS based on EVALS_IN_EPOCH (for initial pool size)')
            self.eval_steps = int(self.dataset.train_dataset.pool_size() / (self.train_batch_size * self.evals_in_epoch))
        self.Factories = utils.factories.Factories(self.prm) # to get hooks

    def train(self):
        super(ClassificationTrainerBase, self).train()
        truth          = self.model.labels
        predictions    = tf.argmax(self.model.predictions, axis=1)
        predictions    = tf.cast(predictions, tf.int32)
        precision      = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

        images_summary = tf.summary.image('images', self.model.images)
        self.summary_writer_train = tf.summary.FileWriter(self.train_dir)  # for training
        self.summary_writer_eval  = tf.summary.FileWriter(self.eval_dir)   # for evaluation

        summary_hook = tf.train.SummarySaverHook(
            save_steps=self.summary_steps,
            summary_writer=self.summary_writer_train,
            summary_op=tf.summary.merge([self.model.summaries,
                                         images_summary,
                                         tf.summary.scalar('Precision', precision)]))

        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': self.model.global_step,
                     'loss_xent': self.model.xent_cost,
                     'loss_wd': self.model.wd_cost,
                     'loss': self.model.cost,
                     'precision': precision},
            every_n_iter=self.logger_steps)

        # LearningRateSetter needs the actual pool size of the trainset to know what is the epoch in every step
        # I assume the pool size is static, otherwise the epoch count becomes meaningless
        learning_rate_hook = self.Factories.get_learning_rate_setter(self.model, self.dataset.train_dataset)
        learning_rate_hook.print_stats() #debug

        self.sess = tf.train.MonitoredTrainingSession(
            checkpoint_dir=self.checkpoint_dir,
            hooks=[logging_hook, learning_rate_hook],
            chief_only_hooks=[summary_hook],
            save_checkpoint_secs=self.checkpoint_secs,
            config=tf.ConfigProto(allow_soft_placement=True))

        self.set_params()

        while not self.sess.should_stop():
            if self.global_step % self.eval_steps == 0 and self._activate_eval:
                self.eval_step()
                self._activate_eval = False
            else:
                self.train_step()
                self._activate_eval = True

    @abstractmethod
    def train_step(self):
        '''Implementing one training step. Must update self.global_step.'''
        pass

    @abstractmethod
    def eval_step(self):
        '''Implementing one evaluation step.'''
        pass

    def print_stats(self):
        super(ClassificationTrainerBase, self).print_stats()
        self.log.info(' EVAL_STEPS: {}'.format(self.eval_steps))
        self.log.info(' EVALS_IN_EPOCH: {}'.format(self.evals_in_epoch))
