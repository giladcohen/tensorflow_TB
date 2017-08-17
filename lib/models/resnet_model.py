from abc import ABCMeta, abstractmethod
from lib.models.classifier_model import ClassifierModel
from lib.models.layers import *
import six


class ResNet(ClassifierModel):
    """Implementing an image classifier using a ResNet architecture
    Related papers:
    https://arxiv.org/pdf/1603.05027v2.pdf
    https://arxiv.org/pdf/1512.03385v1.pdf
    https://arxiv.org/pdf/1605.07146v1.pdf
    """
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self.num_residual_units = self.prm.network.NUM_RESIDUAL_UNITS  # number of residual modules in each unit
        self.num_fc_neurons = 640

    def print_stats(self):
        super(ResNet, self).print_stats()
        self.log.info(' NUM_RESIDUAL_UNITS: {}'.format(self.num_residual_units))

    def _build_inference(self):
        """building the inference model of ResNet"""
        with tf.variable_scope('init'):
            x = tf.map_fn(tf.image.per_image_standardization, self.images)
            x = conv('init_conv', x, 3, 16, stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        filters = [16, 160, 320, 640] #WRN28-10

        with tf.variable_scope('unit_1_0'):
            x = self._residual(x, filters[1], stride_arr(strides[0]), activate_before_residual[0])
        for i in six.moves.range(1, self.num_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = self._residual(x, filters[1], stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
            x = self._residual(x, filters[2], stride_arr(strides[1]), activate_before_residual[1])
        for i in six.moves.range(1, self.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = self._residual(x, filters[2], stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
            x = self._residual(x, filters[3], stride_arr(strides[2]), activate_before_residual[2])
        for i in six.moves.range(1, self.num_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = self._residual(x, filters[3], stride_arr(1), False)

        with tf.variable_scope('unit_last'):
            x = tf.layers.batch_normalization(x, training=self.is_training, name='pre_pool_bn')
            x = relu(x, self.relu_leakiness)
            x = global_avg_pool(x)
            x = self.add_fc_layers(x)
            self.net['pool_out'] = x
            self.logits = fully_connected(x, self.num_classes)

    def _residual(self, x, out_filter, stride, activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        in_filter = x.get_shape().as_list()[3]
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = tf.layers.batch_normalization(x, training=self.is_training, name='init_bn')
                x = relu(x, self.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = tf.layers.batch_normalization(x, training=self.is_training, name='init_bn')
                x = relu(x, self.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = conv('conv1', x, 3, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = tf.layers.batch_normalization(x, training=self.is_training, name='bn2')
            x = relu(x, self.relu_leakiness)
            x = conv('conv2', x, 3, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
            x += orig_x

        self.log.info('image after unit %s', x.get_shape())
        return x

    @abstractmethod
    def add_fc_layers(self, x):
        """Building the fully connected layers of the resnet model.
        Calculating self.logits"""
        pass
