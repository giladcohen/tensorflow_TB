from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer_base import ActiveTrainerBase
import numpy as np
import tensorflow as tf
from lib.base.collections import TRAIN_SUMMARIES
from lib.retention import Retention
from lib.trainers.hooks.global_step_checkpoint_saver_hook import GlobalStepCheckpointSaverHook
from utils.tensorboard_logging import TBLogger


class DynamicModelTrainer(ActiveTrainerBase):

    def train(self):
        while True:
            if self.to_annotate():
                self.annot_step()
                self.update_model()
                self._activate_annot = False
            elif self.to_eval():
                self.eval_step()
                self._activate_eval  = False
            else:
                self.train_step()
                self._activate_annot = True
                self._activate_eval  = True

    def update_model(self):
        pass
