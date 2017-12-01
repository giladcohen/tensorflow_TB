"""
This hook saves checkpoints both after X seconds and at specific global steps
"""

import tensorflow as tf

class GlobalStepCheckpointSaverHook(tf.train.CheckpointSaverHook):

    def __init__(self, steps_to_save, *args, **kwargs):
        super(GlobalStepCheckpointSaverHook, self).__init__(*args, **kwargs)
        if steps_to_save is None:
            steps_to_save = []
        self._steps_to_save = steps_to_save


    def after_run(self, run_context, run_values):
        global_step = run_values.results
        if global_step in self._steps_to_save:
            self._save(global_step, run_context.session)
