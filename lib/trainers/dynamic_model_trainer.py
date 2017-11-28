from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer_base import ActiveTrainerBase
import numpy as np
import tensorflow as tf


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
        lp = self.dataset.train_dataset.pool_size()
        # [16, 160, 320, 640] for 50k samples. for 1k: [2, 4, 6, 14]. weight_decay for 1k: 0.0390625
        if lp == 2000:
            resnet_filters    = np.array([2, 6, 14, 26])
            weight_decay_rate = 0.01953125
        elif lp == 3000:
            resnet_filters = np.array([2, 10, 20, 38])
            weight_decay_rate = 0.01302083333
        elif lp == 4000:
            resnet_filters = np.array([2, 14, 26, 52])
            weight_decay_rate = 0.009765625
        elif lp == 5000:
            resnet_filters = np.array([2, 16, 32, 64])
            weight_decay_rate = 0.0078125
        else:
            err_str = 'pool size is {}. This is not possible'.format(lp)
            self.log.error(err_str)
            raise AssertionError(err_str)
        self.log.info('getting a new model for lp={}. resnet_filters={} and weight_decay_rate={}'
                      .format(lp, resnet_filters, weight_decay_rate))

        tf.reset_default_graph()
        self.model = self.Factories.get_model()
        self.model.resnet_filters = resnet_filters
        self.embedding_dims = resnet_filters[-1]
        self._activate_eval = False  # cannot evaluate fresh model without assigning model variables. train first.
        self.mode = None
        self.sess = None
        self.model.build_graph()
        self.print_model_info()

        # just to re-initialize the graph
        self.sess = self.get_session('train')
        _ = self.sess.run(self.model.global_step, feed_dict=self._get_dummy_feed())

        self.log.info('resetting the global_step to={}'.format(self.global_step))
        self.sess.run(self.model.assign_ops['global_step_ow'], feed_dict={self.model.global_step_ph: self.global_step})

        self.log.info('setting new weight_decay_rate={}'.format(weight_decay_rate))
        self.sess.run(self.model.assign_ops['weight_decay_rate_ow'], feed_dict={self.model.weight_decay_rate_ph: weight_decay_rate})
        self.log.info('Done restoring global_step ({})'.format(self.global_step))
