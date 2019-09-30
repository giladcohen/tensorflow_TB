from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_TB.lib.trainers.active_trainer import ActiveTrainer


class RandomSamplerTrainer(ActiveTrainer):
    def select_new_samples(self):
        return None  # will result in random selection of samples

