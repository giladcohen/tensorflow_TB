from abc import ABCMeta, abstractmethod
from lib.models.model_base import ModelBase
import tensorflow as tf

class ClassifierModelBase(ModelBase):
    __metaclass__ = ABCMeta
    '''Implementing a base image classifier'''

    def __init__(self, *args, **kwargs):
        super(ClassifierModelBase, self).__init__(*args, **kwargs)
        self.num_classes  = self.prm.network.NUM_CLASSES
        self.image_height = self.prm.network.IMAGE_HEIGHT
        self.image_width  = self.prm.network.IMAGE_WIDTH

        self.predictions_prob = None # output of the classifier softmax

    def print_stats(self):
        super(ClassifierModelBase, self).print_stats()
        self.log.info(' NUM_CLASSES: {}'.format(self.num_classes))
        self.log.info(' IMAGE_HEIGHT: {}'.format(self.image_height))
        self.log.info(' IMAGE_WIDTH: {}'.format(self.image_width))

    def _set_placeholders(self):
        super(ClassifierModelBase, self)._set_placeholders()
        self.images = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, 3])
        self.labels = tf.placeholder(tf.int32, [None])

    def _build_interpretation(self):
        '''Interprets the logits'''
        self.predictions_prob = self.calc_prediction_prob()
        self.predictions = tf.argmax(self.predictions_prob, axis=1, output_type=tf.int32)
        self.score       = tf.reduce_mean(tf.to_float(tf.equal(self.predictions, self.labels)))

    @abstractmethod
    def calc_prediction_prob(self):
        """
        :return: The prediction probability vector of the classifier
        """
        pass
