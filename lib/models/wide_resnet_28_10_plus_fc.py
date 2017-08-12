from lib.models.wide_resnet_28_10 import ResNet
from lib.models.layers import *


class WideResNet_28_10_plus_fc(ResNet):

    def __init__(self, *args, **kwargs):
        super(WideResNet_28_10_plus_fc, self).__init__(*args, **kwargs)
        self.num_fc_neurons = self.prm.network.NUM_FC_NEURONS

    def print_stats(self):
        super(ResNet, self).print_stats()
        self.log.info(' NUM_FC_NEURONS: {}'.format(self.num_fc_neurons))

    def add_fc_layers(self, x):
        x = fully_connected(x, self.num_fc_neurons, name='fc_after_pool')
        x = tf.layers.batch_normalization(x, training=self.is_training, name='last_bn')
        x = relu(x, self.relu_leakiness)
        return x
