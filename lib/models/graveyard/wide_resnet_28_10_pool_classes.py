from tensorflow_TB.lib.models.resnet_model import ResNet
from tensorflow_TB.lib.models.layers import *
import tensorflow as tf

class WideResNet_28_10_pool_classes(ResNet):

    def calculate_logits(self, x):
        """generate 5 output neurons for every class - later unite them to a single node for a softmax"""
        x = fully_connected(x, 5 * self.num_classes)
        splits = tf.split(x, num_or_size_splits=self.num_classes, axis=1)
        max_splits = [tf.reduce_max(splits[cls], axis=1, keep_dims=True) for cls in xrange(self.num_classes)]
        logits = tf.concat(max_splits, axis=1)
        return logits
