
"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages



HParams = namedtuple('HParams',
                     'num_classes, lrn_rate, '
                     'num_residual_units, xent_rate, weight_decay_rate, '
                     'relu_leakiness, pool, optimizer')

class ResNet(object):
    """ResNet model."""

    def __init__(self, hps, images, labels, is_training):
        """ResNet constructor.

        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          is_training: True for training. False for evaluation.
        """
        self.hps = hps
        self._images = images
        self.labels = labels
        self.is_training = is_training
        self.net = {}

        self._extra_train_ops = []

    def build_graph(self):
        """Build a whole graph for the model."""
        self._build_inference()
        self._build_interpretation()
        self._build_loss()
        self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def set_params(self):
        self.global_step        = tf.contrib.framework.get_or_create_global_step()
        self.num_classes        = tf.get_variable(
            name='num_classes', dtype=tf.int32, shape=[],
            initializer=tf.constant_initializer(self.hps.num_classes), trainable=False)
        self.lrn_rate           = tf.get_variable(
            name='learning_rate', dtype=tf.float32, shape=[],
            initializer=tf.constant_initializer(self.hps.lrn_rate), trainable=False)
        self.num_residual_units = tf.get_variable(
            name='num_residual_units', dtype=tf.int32, shape=[],
            initializer=tf.constant_initializer(self.hps.num_residual_units), trainable=False)
        self.xent_rate          = tf.get_variable(
            name='xent_rate', dtype=tf.float32, shape=[],
            initializer=tf.constant_initializer(self.hps.xent_rate), trainable=False)
        self.weight_decay_rate  = tf.get_variable(
            name='weight_decay_rate', dtype=tf.float32, shape=[],
            initializer=tf.constant_initializer(self.hps.weight_decay_rate), trainable=False)
        self.relu_leakiness     = tf.get_variable(
            name='relu_leakiness', dtype=tf.float32, shape=[],
            initializer=tf.constant_initializer(self.hps.relu_leakiness), trainable=False)
        self.pool               = tf.get_variable(
            name='pool', dtype=tf.string, shape=[],
            initializer=tf.constant_initializer(self.hps.pool), trainable=False)
        self.optimizer          = tf.get_variable(
            name='optimizer', dtype=tf.string, shape=[],
            initializer=tf.constant_initializer(self.hps.optimizer), trainable=False)

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_inference(self):
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
            x = tf.map_fn(tf.image.per_image_standardization, self._images)
            x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))
            self.net['init_conv'] = x

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual
        filters = [16, 160, 320, 640] #WRN28-10

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                         activate_before_residual[0])
            self.net['unit_1_0_conv'] = x
        for i in six.moves.range(1, self.hps.num_residual_units): #TODO(implement while loop in tf)
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)
                self.net['unit_1_%d_conv' % i] = x

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                         activate_before_residual[1])
            self.net['unit_2_0_conv'] = x
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)
                self.net['unit_2_%d_conv' % i] = x

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                       activate_before_residual[2])
            self.net['unit_3_0_conv'] = x
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)
                self.net['unit_3_%d_conv' % i] = x

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self.relu_leakiness)
            if (self.hps.pool == 'gap'):
                x = self._global_avg_pool(x)
                self.net['pool_out'] = x
            elif (self.hps.pool == 'mp'):
                x = self._max_pool(x)
                self.net['pool_out'] = x
                x = self._fully_connected(x, 32)
                x = self._batch_norm('final_bn2', x)
                x = self._relu(x, self.relu_leakiness)
                self.net['fc1'] = x
            else:
                raise ValueError('illegal value for pool')
            self.logits = self._fully_connected(x, self.hps.num_classes)

    def _build_interpretation(self):
        with tf.variable_scope('interp'):
            self.predictions = tf.nn.softmax(self.logits)

    def _build_loss(self):
        with tf.variable_scope('xent_cost'):
            xent_cost = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            xent_cost = tf.reduce_mean(xent_cost, name='cross_entropy_mean')
            self.xent_cost = tf.multiply(self.xent_rate, xent_cost)
            xent_assert_op = tf.verify_tensor_all_finite(self.xent_cost, 'xent_cost contains NaN or Inf')
            tf.add_to_collection('losses', self.xent_cost)
            tf.add_to_collection('assertions', xent_assert_op)

        with tf.variable_scope('wd_cost'):
            self.wd_cost = self._decay()
            wd_assert_op = tf.verify_tensor_all_finite(self.wd_cost, 'wd_cost contains NaN or Inf')
            tf.add_to_collection('losses', self.wd_cost)
            tf.add_to_collection('assertions', wd_assert_op)

        with tf.control_dependencies(tf.get_collection('assertions')):
                self.cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
                tf.summary.scalar('cost', self.cost)

    def _build_train_op(self):
        """Build training specific ops for the graph."""
        tf.summary.scalar('learning_rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9, use_nesterov=True)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')

        self._extra_train_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            bn = tf.layers.batch_normalization(x, momentum=0.9, training=self.is_training)
            #self._extra_train_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            return bn

    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self.relu_leakiness)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self.relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
              costs.append(tf.nn.l2_loss(var))
              # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.weight_decay_rate, tf.add_n(costs))

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, padding='SAME'):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0/n)))
            return tf.nn.conv2d(x, kernel, strides, padding=padding)

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim, name='fully_connected'):
        """FullyConnected layer for final output."""
        with tf.variable_scope(name):
            x_shape = x.get_shape().as_list()
            input_dim = np.prod(x_shape[1:])
            x = tf.reshape(x, [-1, input_dim])
            w = tf.get_variable(
                'DW', [input_dim, out_dim],
                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            b = tf.get_variable('biases', [out_dim],
                                initializer=tf.constant_initializer())
            return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def _max_pool(self, x, padding='SAME'):
        assert x.get_shape().ndims == 4
        x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding=padding, data_format='NHWC', name='max_pool')
        return x
