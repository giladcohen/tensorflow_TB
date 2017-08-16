from lib.models.wide_resnet_28_10 import ResNet
import tensorflow.contrib.slim as slim
from lib.models.layers import *


class WideResNet_28_10_plus_fc(ResNet):

    def __init__(self, *args, **kwargs):
        super(WideResNet_28_10_plus_fc, self).__init__(*args, **kwargs)
        self.num_fc_neurons = self.prm.network.NUM_FC_NEURONS
        self.normalize_embedding = self.prm.network.NORMALIZE_EMBEDDING

    def print_stats(self):
        super(ResNet, self).print_stats()
        self.log.info(' NUM_FC_NEURONS: {}'.format(self.num_fc_neurons))
        self.log.info(' NORMALIZE_EMBEDDING: {}'.format(self.normalize_embedding))

    def add_fc_layers(self, x):
        x = fully_connected(x, self.num_fc_neurons, name='fc_after_pool')
        x = tf.layers.batch_normalization(x, training=self.is_training, name='last_bn')
        x = relu(x, self.relu_leakiness)
        if self.normalize_embedding:
            x = slim.unit_norm(x, dim=1, scope='normalize_vec')
            variable_summaries(x, 'embedding')
            tf.summary.scalar('x[0]', x[0])
            tf.summary.scalar('x[1]', x[1])
            tf.summary.scalar('x[2]', x[2])
            tf.summary.scalar('x[3]', x[3])
        return x
