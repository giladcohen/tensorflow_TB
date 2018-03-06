from abc import ABCMeta
import tensorflow as tf
from lib.models.resnet_model import ResNet
from lib.tf_alias.metric_loss_ops import cluster_loss, lifted_struct_loss, compute_clustering_score
from lib.base.collections import LOSSES

class DMLResNet(ResNet):
    __metaclass__ = ABCMeta
    '''Implementing an image classifier using DML'''

    def __init__(self, *args, **kwargs):
        super(DMLResNet, self).__init__(*args, **kwargs)
        self.cluster_cost = None     # contribution of the lifted structured feature embedding loss

    def print_stats(self):
        super(DMLResNet, self).print_stats()
        self.log.info(' DML_MARGIN_MULTIPLIER: {}'.format(self.prm.network.optimization.DML_MARGIN_MULTIPLIER))

    def _init_params(self):
        super(DMLResNet, self)._init_params()
        self.dml_margin_multiplier           = tf.contrib.framework.model_variable(
            name='dml_margin_multiplier', dtype=tf.float32, shape=[],
            initializer=tf.constant_initializer(self.prm.network.optimization.DML_MARGIN_MULTIPLIER), trainable=False)

    def _set_params(self):
        super(DMLResNet, self)._set_params()
        self.assign_ops['dml_margin_multiplier']    = self.dml_margin_multiplier.assign(self.prm.network.optimization.DML_MARGIN_MULTIPLIER)
        self.assign_ops['dml_margin_multiplier_ow'] = self.dml_margin_multiplier.assign(self.dml_margin_multiplier_ph)

    def _set_placeholders(self):
        super(DMLResNet, self)._set_placeholders()
        self.dml_margin_multiplier_ph = tf.placeholder(tf.float32)

    def add_fidelity_loss(self):
        with tf.variable_scope('cluster_cost'):
            # labels_expanded = tf.expand_dims(self.labels, axis=-1, name='labels_expanded')
            # cluster_cost, self.score = cluster_loss(
            #     labels=labels_expanded,
            #     embeddings=self.net['embedding_layer'],
            #     margin_multiplier=self.dml_margin_multiplier)
            cluster_cost = lifted_struct_loss(self.labels, self.net['embedding_layer'], margin=self.dml_margin_multiplier)
            self.cluster_cost = tf.multiply(self.xent_rate, cluster_cost)  #TODO(gilad): think of better name here (not xent_rate)
            tf.summary.scalar('cluster_cost', self.cluster_cost)
            cluster_assert_op = tf.verify_tensor_all_finite(self.cluster_cost, 'cluster_cost contains NaN or Inf')
            tf.add_to_collection(LOSSES, self.cluster_cost)
            tf.add_to_collection('assertions', cluster_assert_op)

    def _build_interpretation(self):
        '''Interprets the logits'''
        self.score = self.cluster_cost

    def calculate_logits(self, x):
        return None
