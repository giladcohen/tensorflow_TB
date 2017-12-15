"""
This hook saves summaries after X steps, and only after training (not for evaluation or prediction)
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import lib.logger.logger as logger

class TrainSummarySaverHook(tf.train.SummarySaverHook):

    def __init__(self, name, prm, model, *args, **kwargs):
        super(TrainSummarySaverHook, self).__init__(*args, **kwargs)
        self.name = name
        self.prm = prm
        self.log = logger.get_logger(name)
        self.model = model

    def __str__(self):
        return self.name

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._request_summary = (
            self._next_step is None or
            self._timer.should_trigger_for_step(self._next_step))
        requests = {"global_step": self.model.global_step,
                    "is_training": self.model.is_training}
        if self._request_summary:
          if self._get_summary_op() is not None:
            requests["summary"] = self._get_summary_op()

        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        is_training = run_values.results["is_training"]
        if not is_training:
            return
        super(TrainSummarySaverHook, self).after_run(self, run_context, run_values)

