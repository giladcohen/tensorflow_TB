from lib.models.resnet_model import ResNet
from lib.models.layers import *

class WideResNet_28_10_plus_fc(ResNet):

    def __init__(self, *args, **kwargs):
        super(WideResNet_28_10_plus_fc, self).__init__(*args, **kwargs)
        self.batch_normalize_embedding = self.prm.network.BATCH_NORMALIZE_EMBEDDING

    def print_stats(self):
        super(ResNet, self).print_stats()
        self.log.info(' BATCH_NORMALIZE_EMBEDDING: {}'.format(self.batch_normalize_embedding))

    def post_pool_operations(self, x):
        x = fully_connected(x, self.embedding_dims, name='fc_after_pool')
        if self.batch_normalize_embedding:
            x = tf.layers.batch_normalization(x, training=self.is_training, name='last_bn')
        x = relu(x, self.relu_leakiness)
        return x
