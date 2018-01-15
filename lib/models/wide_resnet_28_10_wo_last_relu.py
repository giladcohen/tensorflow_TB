from lib.models.resnet_model import ResNet
from lib.models.layers import *
import tensorflow as tf

class WideResNet_28_10_wo_last_relu(ResNet):

    def unit_last(self, x):
        """Implementing the final unit of the resnet"""
        with tf.variable_scope('unit_last'):
            x = tf.layers.batch_normalization(x, training=self.is_training, name='pre_pool_bn')
            # x = relu(x, self.relu_leakiness)
            x = global_avg_pool(x)
            x = self.post_pool_operations(x)
            x = tf.nn.dropout(x, keep_prob=self.dropout_keep_prob)
            if self.normalize_embedding:
                x = tf.nn.l2_normalize(x, axis=1, name='normalize_vec')  # was x = slim.unit_norm(x, dim=1, scope='normalize_vec')
                variable_summaries('embedding', x)
            self.net['embedding_layer'] = x
            self.net['logits'] = self.calculate_logits(x)
