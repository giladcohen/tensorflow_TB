from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.base.collections import TRAIN_SUMMARIES
from lib.trainers.classification_trainer import ClassificationTrainer

class DMLClassificationTrainer(ClassificationTrainer):
    """Implementing classification trainer for DML
    Using the entire labeled trainset for training"""


    def eval_step(self):
        '''Implementing one evaluation step.'''
        pass  # FIXME(gilad): Implement KNN score

    def get_train_summaries(self):
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.scalar('score', self.model.score))  # FIXME(gilad) : implement KNN score
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.scalar('weight_decay_rate', self.model.weight_decay_rate))
