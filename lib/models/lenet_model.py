from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from lib.models.classifier_model import ClassifierModel
import tensorflow as tf
from lib.models.layers import variable_summaries
# from nets import lenet

slim = tf.contrib.slim

class LeNet(ClassifierModel):
    """
    Implementing an image classifier using a variant on the simple LeNet network
    """

    def _build_inference(self):
        with tf.variable_scope('LeNet'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                 weights_regularizer=slim.l2_regularizer(self.weight_decay_rate),
                                 weights_initializer=tf.glorot_uniform_initializer(seed=self.prm.SUPERSEED),
                                 activation_fn=tf.nn.relu):
                x = slim.conv2d(self.images, 32, [5, 5], scope='conv1')
                x = slim.max_pool2d(x, [2, 2], 2, scope='pool1')
                x = slim.conv2d(x, 64, [5, 5], scope='conv2')
                x = slim.max_pool2d(x, [2, 2], 2, scope='pool2')
                x = slim.flatten(x)
                x = slim.fully_connected(x, self.embedding_dims, scope='fc3')
                x = tf.nn.dropout(x, keep_prob=self.dropout_keep_prob)
                if self.normalize_embedding:
                    x = tf.nn.l2_normalize(x, axis=1, name='normalize_vec')
                variable_summaries('embedding', x)
                self.net['embedding_layer'] = x
                self.net['logits'] = slim.fully_connected(x, self.num_classes, activation_fn=None, scope='fc4')

    def _decay(self):
        """L2 weight decay loss."""
        wd_cost = slim.losses.get_regularization_losses()
        return tf.multiply(self.weight_decay_rate, wd_cost)
