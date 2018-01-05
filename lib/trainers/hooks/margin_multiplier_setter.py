from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import lib.logger.logger as logger

class MarginMultiplierSetter(tf.train.SessionRunHook):
    """Sets DML margin multiplier based on the NMI score."""

    def __init__(self, name, prm, model, retention):
        self.name = name
        self.prm = prm
        self.model = model
        self.retention = retention
        self.log = logger.get_logger(name)

        self._init_mm                  = self.prm.network.optimization.DML_MARGIN_MULTIPLIER
        self.decay_refractory_steps    = self.prm.train.train_control.margin_multiplier_setter.MM_DECAY_REFRACTORY_STEPS
        self.global_step_of_last_decay = 0

        self._mm = self._init_mm

    def __str__(self):
        return self.name

    def before_run(self, run_context):
        requests = {"global_step": self.model.global_step}

        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        """Check if the NMI score is stuck. If it is, reassign the value of the DML margin multiplier"""
        global_step = run_values.results['global_step']

        if global_step - self.global_step_of_last_decay < self.decay_refractory_steps:
            # if we didn't wait decay_refractory_steps number of steps from the last decay, do nothing
            return
        if self.retention.is_score_stuck():
            self.log.info('global_step={}: Validtion score did not improve after {} steps. Decreasing the margin multiplier by a factor of {} to {}. Last margin multiplier decay was before {} steps.' \
                          .format(global_step, global_step - self.retention.best_score_step, 0.94, 0.94 * self._mm, global_step - self.global_step_of_last_decay))
            self.set_margin_multiplier(0.94 * self._mm, run_context.session)
            self.global_step_of_last_decay = global_step

    def get_margin_multiplier(self):
        return self._mm

    def set_margin_multiplier(self, mm, sess):
        """Setting the margin multiplier. We need to assign a variable in the graph, so we need a session as well"""
        self.log.info('set_margin_multiplier: changing the dml margin multiplier from {} to {}'.format(self._mm, mm))
        self._mm = mm
        sess.run(self.model.assign_ops['dml_margin_multiplier_ow'],
                 feed_dict={self.model.dml_margin_multiplier_ph: self._mm})

    def reset_margin_multiplier(self, sess):
        self.log.info('Reseting margin multiplier to initial value')
        self.set_margin_multiplier(self._init_mm, sess)

    def print_stats(self):
        self.log.info(' MM_DECAY_REFRACTORY_STEPS: {}'.format(self.decay_refractory_steps))
