from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.dynamic_model_trainer import DynamicModelTrainer


class RandomSamplerDynamicTrainer(DynamicModelTrainer):
    def select_new_samples(self):
        return None  # will result in random selection of samples
