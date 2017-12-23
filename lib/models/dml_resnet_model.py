from abc import ABCMeta
import tensorflow as tf
from lib.models.resnet_model import ResNet
from tensorflow.contrib.losses.python.metric_learning import cluster_loss

class DMLResNet(ResNet):
    __metaclass__ = ABCMeta
    '''Implementing an image classifier using DML'''

    def __init__(self, *args, **kwargs):
        super(DMLResNet, self).__init__(*args, **kwargs)
        self.dml_margin   = self.prm.network.optimization.DML_MARGIN_MULTIPLIER

        self.cluster_cost = None     # contribution of the lifted structured feature embedding loss

    def print_stats(self):
        super(DMLResNet, self).print_stats()
        self.log.info(' DML_MARGIN: {}'.format(self.dml_margin))

    def add_fidelity_loss(self):
        with tf.variable_scope('cluster_cost'):
            labels_expanded = tf.expand_dims(self.labels, axis=-1, name='labels_expanded')
            cluster_cost = cluster_loss(labels=labels_expanded,
                                        embeddings=self.net['embedding_layer'],
                                        margin_multiplier=self.dml_margin)
            self.cluster_cost = tf.multiply(self.xent_rate, cluster_cost)  #TODO(gilad): think of better name here (not xent_rate)
            tf.summary.scalar('cluster_cost', self.cluster_cost)
            cluster_assert_op = tf.verify_tensor_all_finite(self.cluster_cost, 'cluster_cost contains NaN or Inf')
            tf.add_to_collection(tf.GraphKeys.LOSSES, self.cluster_cost)
            tf.add_to_collection('assertions', cluster_assert_op)

    def _build_interpretation(self):
        self.score = self.cluster_cost

    def calculate_logits(self, x):
        return None
