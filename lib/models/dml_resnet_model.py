from abc import ABCMeta
import tensorflow as tf
from lib.models.resnet_model import ResNet
#from tensorflow.contrib.losses.python.metric_learning import cluster_loss
from lib.tf_alias.metric_loss_ops import cluster_loss
from utils.misc import get_vars

class DMLResNet(ResNet):
    __metaclass__ = ABCMeta
    '''Implementing an image classifier using DML'''

    def __init__(self, *args, **kwargs):
        super(DMLResNet, self).__init__(*args, **kwargs)
        self.dml_margin_multiplier   = self.prm.network.optimization.DML_MARGIN_MULTIPLIER

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
        self.assign_ops['dml_margin_multiplier'] = self.dml_margin_multiplier.assign(self.prm.network.optimization.DML_MARGIN_MULTIPLIER)

    def add_fidelity_loss(self):
        with tf.variable_scope('cluster_cost'):
            labels_expanded = tf.expand_dims(self.labels, axis=-1, name='labels_expanded')
            cluster_cost, _ = cluster_loss(labels=labels_expanded,
                                           embeddings=self.net['embedding_layer'],
                                           margin_multiplier=self.dml_margin_multiplier)
            self.cluster_cost = tf.multiply(self.xent_rate, cluster_cost)  #TODO(gilad): think of better name here (not xent_rate)
            tf.summary.scalar('cluster_cost', self.cluster_cost)
            cluster_assert_op = tf.verify_tensor_all_finite(self.cluster_cost, 'cluster_cost contains NaN or Inf')
            tf.add_to_collection(tf.GraphKeys.LOSSES, self.cluster_cost)
            tf.add_to_collection('assertions', cluster_assert_op)

    def _build_interpretation(self):
        labels_expanded = tf.expand_dims(self.labels, axis=-1, name='labels_expanded')
        _, self.score = cluster_loss(labels=labels_expanded,
                                     embeddings=self.net['embedding_layer'],
                                     margin_multiplier=self.dml_margin_multiplier)

    def calculate_logits(self, x):
        return None

    # def _build_train_op(self):
    #     """Build trainers specific ops for the graph."""
    #     tf.summary.scalar('learning_rate', self.lrn_rate)
    #     vars_to_train, vars_to_ignore = get_vars(tf.trainable_variables(),
    #                                              'unit_3_3',
    #                                              'unit_last')
    #     trainable_variables = vars_to_train
    #
    #     grads = tf.gradients(self.cost, trainable_variables)
    #     optimizer = self._get_optimizer()
    #
    #     apply_op = optimizer.apply_gradients(
    #         zip(grads, trainable_variables),
    #         global_step=self.global_step, name='train_step')
    #
    #     self._extra_train_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    #     train_ops = [apply_op] + self._extra_train_ops
    #     self.train_op = tf.group(*train_ops)
