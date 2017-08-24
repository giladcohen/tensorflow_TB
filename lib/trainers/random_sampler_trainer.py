from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer_base import ActiveTrainerBase


class RandomSamplerTrainer(ActiveTrainerBase):
    def select_new_samples(self):
        self.dataset.train_dataset.update_pool()

