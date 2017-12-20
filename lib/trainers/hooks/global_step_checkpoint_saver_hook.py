"""
This hook saves checkpoints at specific global steps
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import os
import lib.logger.logger as logger

from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.core.util.event_pb2 import SessionLog

class GlobalStepCheckpointSaverHook(tf.train.SessionRunHook):

    def __init__(self, name, prm, model, steps_to_save, checkpoint_dir, saver, checkpoint_basename='model_schedule.ckpt'):
        self.name = name
        self.prm = prm
        self.log = logger.get_logger(name)
        self.model = model  # model might change between runs, cannot use global train step. Must use model step.
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)

        if steps_to_save is None:
            steps_to_save = []
        self._steps_to_save = steps_to_save

    def __str__(self):
        return self.name

    def begin(self):
        self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self.model.global_step, self.model.is_training])  # Asks for global step and whether or not we are training

    def after_run(self, run_context, run_values):
        global_step = run_values.results[0]
        is_training = run_values.results[1]
        if global_step in self._steps_to_save and is_training:
            self._save(run_context.session, global_step)

    def _save(self, session, step):
        """Saves the latest checkpoint."""
        self.log.info("Saving checkpoints for %d into %s.", step, self._save_path)

        self._saver.save(session, self._save_path, global_step=step)
        self._summary_writer.add_session_log(
            SessionLog(
                status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
            step)
