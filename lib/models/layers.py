import tensorflow as tf
import numpy as np

def stride_arr(stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

def conv(name, x, filter_size, out_filters, strides, padding='SAME'):
    """Convolution."""
    with tf.variable_scope(name):
        in_filters = x.get_shape().as_list()[3]
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0/n)))
        conv_out = tf.nn.conv2d(x, kernel, strides, padding=padding)
        return conv_out

def relu(x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

def global_avg_pool(x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

def fully_connected(x, out_dim, name='fully_connected'):
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
