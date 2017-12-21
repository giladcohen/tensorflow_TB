from abc import ABCMeta
from lib.models.classifier_model_base import ClassifierModelBase
import tensorflow as tf

class ClassifierModel(ClassifierModelBase):
    __metaclass__ = ABCMeta
    '''Implementing an image classifier using softmax with cross entropy'''

    def __init__(self, *args, **kwargs):
        super(ClassifierModel, self).__init__(*args, **kwargs)
        self.xent_cost    = None     # contribution of cross entropy to loss

    def calc_prediction_prob(self):
        return tf.nn.softmax(self.net['logits'])

    def add_fidelity_loss(self):
        with tf.variable_scope('xent_cost'):
            xent_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.net['logits'], labels=self.labels)
            xent_cost = tf.reduce_mean(xent_cost, name='cross_entropy_mean')
            self.xent_cost = tf.multiply(self.xent_rate, xent_cost)
            tf.summary.scalar('xent_cost', self.xent_cost)
            xent_assert_op = tf.verify_tensor_all_finite(self.xent_cost, 'xent_cost contains NaN or Inf')
            tf.add_to_collection(tf.GraphKeys.LOSSES, self.xent_cost)
            tf.add_to_collection('assertions', xent_assert_op)
