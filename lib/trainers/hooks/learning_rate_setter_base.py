from __future__ import division

import tensorflow as tf
import lib.logger.logger as logger

class LearningRateSetterBase(tf.train.SessionRunHook):
    """Sets learning_rate based on the initial learning rate parameter."""

    def __init__(self, name, prm, model, trainset_dataset):
        self.name = name
        self.prm = prm
        self.model = model
        self.trainset_dataset = trainset_dataset  # used in children
        self.log = logger.get_logger(name)

        self.learning_rate_setter  = self.prm.train.train_control.learning_rate_setter.LEARNING_RATE_SETTER
        self._init_lrn_rate = self.prm.network.optimization.LEARNING_RATE

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
