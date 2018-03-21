from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from lib.models.classifier_model import ClassifierModel
import tensorflow as tf
from lib.models.layers import *
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
                self.net['input_images'] = self.images
                x = slim.conv2d(self.images, 32, [5, 5], scope='conv1')
                self.net['conv1'] = x
                x = slim.max_pool2d(x, [2, 2], 2, scope='pool1')
                self.net['pool1'] = x
                x = slim.conv2d(x, 64, [5, 5], scope='conv2')
                self.net['conv2'] = x
                x = slim.max_pool2d(x, [2, 2], 2, scope='pool2')
                self.net['pool2'] = x
                x = slim.flatten(x)
                x = slim.fully_connected(x, self.embedding_dims, scope='fc3')
                x = tf.nn.dropout(x, keep_prob=self.dropout_keep_prob)
                if self.normalize_embedding:
                    x = tf.nn.l2_normalize(x, axis=1, name='normalize_vec')
                variable_summaries('embedding', x)
                self.net['embedding_layer'] = x
                self.net['logits'] = slim.fully_connected(x, self.num_classes, activation_fn=None, scope='fc4')

                for key in self.net.keys():
                    self.net[key + '_gap'] = global_avg_pool(self.net[key])

    def _decay(self):
        """L2 weight decay loss."""
        costs = tf.losses.get_regularization_losses()
        return tf.multiply(self.weight_decay_rate, tf.add_n(costs))
