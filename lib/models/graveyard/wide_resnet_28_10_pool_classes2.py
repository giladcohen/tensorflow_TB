"""
This model is similar to WideResNet_28_10_pool_classes except that here I am pooing the softmax output
instead of a class' logits.
"""

from lib.models.resnet_model import ResNet
from lib.models.layers import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.base.collections import LOSSES


class WideResNet_28_10_pool_classes2(ResNet):

    def calculate_logits(self, x):
        """generate 5 output neurons for every class - later unite them to a single node for a softmax"""
        logits = fully_connected(x, 5 * self.num_classes)
        return logits

    def _build_interpretation(self):
        '''Interprets the logits'''
        softmax_out = tf.nn.softmax(self.net['logits'])  # 5 * num_classes vector
        splits = tf.split(softmax_out, num_or_size_splits=self.num_classes, axis=1)
        max_splits = [tf.reduce_max(splits[cls], axis=1, keep_dims=True) for cls in xrange(self.num_classes)]
        softmax_out_pooled = tf.concat(max_splits, axis=1)
        softmax_out_norm = slim.unit_norm(softmax_out_pooled, dim=1, scope='normalize_softmax')

        self.predictions_prob = softmax_out_norm
        self.predictions = tf.argmax(self.predictions_prob, axis=1, output_type=tf.int32)
        self.score       = tf.reduce_mean(tf.to_float(tf.equal(self.predictions, self.labels)))

    def add_fidelity_loss(self):
        with tf.variable_scope('xent_cost'):
            labels_one_hot = tf.one_hot(self.labels, depth=self.num_classes, on_value=1, off_value=0)
            xent_cost = tf.losses.log_loss(labels=labels_one_hot, predictions=self.predictions_prob)
            xent_cost = tf.reduce_mean(xent_cost, name='cross_entropy_mean')
            self.xent_cost = tf.multiply(self.xent_rate, xent_cost)
            xent_assert_op = tf.verify_tensor_all_finite(self.xent_cost, 'xent_cost contains NaN or Inf')
            tf.add_to_collection(LOSSES, self.xent_cost)
            tf.add_to_collection('assertions', xent_assert_op)
