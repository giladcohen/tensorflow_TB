from __future__ import division

import tensorflow as tf
import lib.logger.logger as logger

class LearningRateSetterBase(tf.train.SessionRunHook):
    """Sets learning_rate based on the initial learning rate parameter."""

    def __init__(self, name, prm, model, trainset_dataset, precision_retention):
        self.name = name
        self.prm = prm
        self.model = model
        self.trainset_dataset = trainset_dataset  # used in children
        self.precision_retention = precision_retention  # used in children
        self.log = logger.get_logger(name)

        self.learning_rate_setter  = self.prm.train.train_control.learning_rate_setter.LEARNING_RATE_SETTER
        self._init_lrn_rate  = self.prm.network.optimization.LEARNING_RATE
        self._reset_lrn_rate = self.prm.train.train_control.learning_rate_setter.LEARNING_RATE_RESET
        if self._reset_lrn_rate is None:
            self.log.warning('LEARNING_RATE_RESET is None. Setting LEARNING_RATE_RESET=LEARNING_RATE')
            self._reset_lrn_rate = self.prm.network.optimization.LEARNING_RATE

    def __str__(self):
        return self.name

    def begin(self):
        self._lrn_rate = self._init_lrn_rate

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            self.model.global_step,  # Asks for global step value.
            feed_dict={self.model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def print_stats(self):
        self.log.info('Learning Rate Setter parameters:')
        self.log.info(' LEARNING_RATE_SETTER: {}'.format(self.learning_rate_setter))
        self.log.info(' LEARNING_RATE_RESET: {}'.format(self._reset_lrn_rate))

    def get_lrn_rate(self):
        return self._lrn_rate

    def set_lrn_rate(self, lrn_rate):
        self.log.info('set_lrn_rate: changing the learning rate from {} to {}'.format(self._lrn_rate, lrn_rate))
        self._lrn_rate = lrn_rate

    def reset_learning_rate(self):
        self.log.info('Reseting learning rate to reset value')
        self.set_lrn_rate(self._reset_lrn_rate)
