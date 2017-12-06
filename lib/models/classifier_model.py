from abc import ABCMeta
from lib.models.model_base import ModelBase
import tensorflow as tf

class ClassifierModel(ModelBase):
    __metaclass__ = ABCMeta
    '''Implementing an image classifier'''

    def __init__(self, *args, **kwargs):
        super(ClassifierModel, self).__init__(*args, **kwargs)
        self.num_classes  = self.prm.network.NUM_CLASSES
        self.image_height = self.prm.network.IMAGE_HEIGHT
        self.image_width  = self.prm.network.IMAGE_WIDTH
        self.xent_cost    = None     # contribution of cross entropy to loss
        self.predictions_prob = None # output of the classifier softmax

    def print_stats(self):
        super(ClassifierModel, self).print_stats()
        self.log.info(' NUM_CLASSES: {}'.format(self.num_classes))
        self.log.info(' IMAGE_HEIGHT: {}'.format(self.image_height))
        self.log.info(' IMAGE_WIDTH: {}'.format(self.image_width))

    def _set_placeholders(self):
        super(ClassifierModel, self)._set_placeholders()
        self.images = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, 3])
        self.labels = tf.placeholder(tf.int32, [None])

    def _build_interpretation(self):
        '''Interprets the logits'''
        self.predictions_prob = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.predictions_prob, axis=1, output_type=tf.int32)
        self.score       = tf.reduce_mean(tf.to_float(tf.equal(self.predictions, self.labels)))

    def add_fidelity_loss(self):
        with tf.variable_scope('xent_cost'):
            xent_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            xent_cost = tf.reduce_mean(xent_cost, name='cross_entropy_mean')
            self.xent_cost = tf.multiply(self.xent_rate, xent_cost)
            xent_assert_op = tf.verify_tensor_all_finite(self.xent_cost, 'xent_cost contains NaN or Inf')
            tf.add_to_collection('losses', self.xent_cost)
            tf.add_to_collection('assertions', xent_assert_op)
