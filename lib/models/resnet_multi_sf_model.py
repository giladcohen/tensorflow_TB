from tensorflow_TB.lib.models.resnet_model import ResNet
from tensorflow_TB.lib.models.layers import *
import six
from tensorflow_TB.lib.base.collections import LOSSES

class ResnetMultiSf(ResNet):
    def __init__(self, *args, **kwargs):
        super(ResnetMultiSf, self).__init__(*args, **kwargs)
        self.multi_sf = self.prm.network.MULTI_SF

        self.sf_layer_list = ['unit_1_3', 'unit_2_3', 'unit_3_2']

        if self.multi_sf:
            self.sf_wight_dict = {
                'unit_1_0': 0.0,
                'unit_1_1': 0.0,
                'unit_1_2': 0.0,
                'unit_1_3': 0.125,
                'unit_2_0': 0.0,
                'unit_2_1': 0.0,
                'unit_2_2': 0.0,
                'unit_2_3': 0.25,
                'unit_3_0': 0.0,
                'unit_3_1': 0.0,
                'unit_3_2': 0.5,
                'unit_3_3': 0.0
            }
        else:
            self.sf_wight_dict = {
                'unit_1_0': 0.0,
                'unit_1_1': 0.0,
                'unit_1_2': 0.0,
                'unit_1_3': 0.0,
                'unit_2_0': 0.0,
                'unit_2_1': 0.0,
                'unit_2_2': 0.0,
                'unit_2_3': 0.0,
                'unit_3_0': 0.0,
                'unit_3_1': 0.0,
                'unit_3_2': 0.0,
                'unit_3_3': 0.0
            }

    def print_stats(self):
        super(ResnetMultiSf, self).print_stats()
        self.log.info(' MULTI_SF: {}'.format(self.multi_sf))

    def _build_inference(self):
        """Build mini FC networks for units: unit_1_3, unit_2_3, and unit_3_2"""
        super(ResnetMultiSf, self)._build_inference()
        for layer in self.sf_layer_list:
            with tf.variable_scope(layer):
                self.net[layer + '_logits'] = self.calculate_logits(self.net[layer + '_relu_gap'])

    def _build_loss(self):
        self.add_multi_sf_loss()
        super(ResnetMultiSf, self)._build_loss()

    def add_multi_sf_loss(self):
        """
        Adding softmax lossed for the output of the layers listed in sf_weight_dict, after applying
        global average pooling on those layers' outputs.
        :return: None
        """
        for layer in self.sf_layer_list:
            xent_cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.net[layer + '_logits'], labels=self.labels)
            xent_cost = tf.reduce_mean(xent_cost, name=layer + 'cross_entropy_mean')
            xent_cost = tf.multiply(self.sf_wight_dict[layer], xent_cost)
            tf.summary.scalar('xent_cost/' + layer, xent_cost)
            xent_assert_op = tf.verify_tensor_all_finite(xent_cost, 'xent_cost/{} contains NaN or Inf'.format(layer))
            tf.add_to_collection(LOSSES, xent_cost)
            tf.add_to_collection('assertions', xent_assert_op)
