from __future__ import division
import tensorflow as tf
import lib.logger.logger as logger

class LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def __init__(self, name, prm, model):
        self.name = name
        self.prm = prm
        self.model = model
        self.log = logger.get_logger(name)
        self.train_batch_size = self.prm.train.train_control.TRAIN_BATCH_SIZE
        self.epoch_size = self.prm.dataset.TRAIN_SET_SIZE
        self._init_lrn_rate = self.prm.network.optimization.LEARNING_RATE
        self._notify = [False, False, False, False]

    def begin(self):
        self._lrn_rate = self._init_lrn_rate

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            self.model.global_step,  # Asks for global step value.
            feed_dict={self.model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
        train_step = run_values.results
        epoch = (self.train_batch_size * train_step) // self.epoch_size
        if epoch < 60:
            self._lrn_rate = self._init_lrn_rate
            if not self._notify[0]:
                self.log.info('epoch={}. Decreasing learning rate to {}'.format(epoch, self._lrn_rate))
                self._notify[0] = True
        elif epoch < 120:
            self._lrn_rate = self._init_lrn_rate/5
            if not self._notify[1]:
                self.log.info('epoch={}. Decreasing learning rate to {}'.format(epoch, self._lrn_rate))
                self._notify[1] = True
        elif epoch < 160:
            self._lrn_rate = self._init_lrn_rate/25
            if not self._notify[2]:
                self.log.info('epoch={}. Decreasing learning rate to {}'.format(epoch, self._lrn_rate))
                self._notify[2] = True
        else:
            self._lrn_rate = self._init_lrn_rate/125
            if not self._notify[3]:
                self.log.info('epoch={}. Decreasing learning rate to {}'.format(epoch, self._lrn_rate))
                self._notify[3] = True

