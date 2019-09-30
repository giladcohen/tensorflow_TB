from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from tensorflow_TB.lib.models.classifier_model import ClassifierModel
import tensorflow as tf
from tensorflow_TB.lib.models.layers import variable_summaries

slim = tf.contrib.slim

class FC2Net(ClassifierModel):
    """
    Implementing an image classifier using a very simple two fully connected layers
    """

    def _build_inference(self):
        with tf.variable_scope('FC2Net'):
            with slim.arg_scope([slim.fully_connected],
                                 weights_regularizer=slim.l2_regularizer(self.weight_decay_rate),
                                 weights_initializer=tf.glorot_uniform_initializer(seed=self.prm.SUPERSEED),
                                 activation_fn=tf.nn.relu):
                self.net['input_images'] = self.images
                x = slim.flatten(self.images)
                x = slim.fully_connected(x, self.embedding_dims, scope='fc1')
                x = tf.nn.dropout(x, keep_prob=self.dropout_keep_prob)
                if self.normalize_embedding:
                    x = tf.nn.l2_normalize(x, axis=1, name='normalize_vec')
                variable_summaries('embedding', x)
                self.net['embedding_layer'] = x
                self.net['logits'] = slim.fully_connected(x, self.num_classes, activation_fn=None, scope='fc2')

    def _decay(self):
        """L2 weight decay loss."""
        costs = tf.losses.get_regularization_losses()
        return tf.multiply(self.weight_decay_rate, tf.add_n(costs))
