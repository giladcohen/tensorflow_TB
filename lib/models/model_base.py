
"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import lib.logger.logger as logger


class ModelBase(object):
    __metaclass__ = ABCMeta
    """base model class for NNs"""

    def __init__(self, name, prm):
        """model constructor.
        Args:
            name: name of model
            prm: parameters
        """
        self.name = name
        self._logger = logger.get_logger(name)
        self.prm = prm
        self.train_op = None       # training operation
        self.cost = None           # total objective to decrease - input to train_op
        self.wd_cost = None        # weight decay cost
        self.logits = None         # output of network - used to calculate cost
        self.summaries = None      # summaries collected from the entire graph
        self.net = {}              # optional hash map for sampling signals along the network
        self.assign_ops = {}       # optional assign operations
        self._extra_train_ops = [] # optional training operations to apply

    def __str__(self):
        return self.prm.network.ARCHITECTURE

    def build_graph(self):
        """Build a whole graph for the model."""
        self._init_params()
        self._set_params()
        with tf.variable_scope('placeholders'):
            self._set_placeholders()
        with tf.variable_scope('inference'):
            self._build_inference()
        with tf.variable_scope('interpretation'):
            self._build_interpretation()
        with tf.variable_scope('loss'):
            self._build_loss()
        with tf.variable_scope('training'):
            self._build_train_op()
        with tf.variable_scope('summaries'):
            self.summaries = tf.summary.merge_all()

    def _init_params(self):
        self.global_step        = tf.contrib.framework.get_or_create_global_step()
        self.lrn_rate           = tf.contrib.framework.model_variable(
            name='learning_rate', dtype=tf.float32, shape=[],
            initializer=tf.constant_initializer(self.prm.network.optimization.LEARNING_RATE), trainable=False)
        self.xent_rate          = tf.contrib.framework.model_variable(
            name='xentropy_rate', dtype=tf.float32, shape=[],
            initializer=tf.constant_initializer(self.prm.network.optimization.XENTROPY_RATE), trainable=False)
        self.weight_decay_rate  = tf.contrib.framework.model_variable(
            name='weight_decay_rate', dtype=tf.float32, shape=[],
            initializer=tf.constant_initializer(self.prm.network.optimization.WEIGHT_DECAY_RATE), trainable=False)
        self.relu_leakiness     = tf.contrib.framework.model_variable(
            name='relu_leakiness', dtype=tf.float32, shape=[],
            initializer=tf.constant_initializer(self.prm.network.system.RELU_LEAKINESS), trainable=False)
        self.optimizer          = tf.contrib.framework.model_variable(
            name='optimizer', dtype=tf.string, shape=[],
            initializer=tf.constant_initializer(self.prm.network.optimization.OPTIMIZER), trainable=False)

    def _set_params(self):
        self.assign_ops['lrn_rate'] = self.lrn_rate.assign(self.prm.network.optimization.LEARNING_RATE)
        self.assign_ops['xent_rate'] = self.xent_rate.assign(self.prm.network.optimization.XENTROPY_RATE)
        self.assign_ops['weight_decay_rate'] = self.weight_decay_rate.assign(self.prm.network.optimization.WEIGHT_DECAY_RATE)
        self.assign_ops['relu_leakiness'] = self.relu_leakiness.assign(self.prm.network.system.RELU_LEAKINESS)
        self.assign_ops['optimizer'] = self.optimizer.assign(self.prm.network.optimization.OPTIMIZER)

    @abstractmethod
    def _set_placeholders(self):
        '''Setting up inputs for the network'''
        pass

    @abstractmethod
    def _build_inference(self):
        '''build the inference model and sets self.logits'''
        pass

    def _build_interpretation(self):
        '''Interprets the logits'''
        pass

    def _build_loss(self):
        '''calculate self.cost for train_op'''
        self.add_weight_decay()
        self.add_fidelity_loss()
        with tf.control_dependencies(tf.get_collection('assertions')):
            self.cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
            tf.summary.scalar('cost', self.cost)

    def add_weight_decay(self):
        with tf.variable_scope('wd_cost'):
            self.wd_cost = self._decay()
            wd_assert_op = tf.verify_tensor_all_finite(self.wd_cost, 'wd_cost contains NaN or Inf')
            tf.add_to_collection('losses', self.wd_cost)
            tf.add_to_collection('assertions', wd_assert_op)

    @abstractmethod
    def add_fidelity_loss(self):
        """Add fidelity loss to self.cost"""
        pass

    def _build_train_op(self):
        """Build trainers specific ops for the graph."""
        tf.summary.scalar('learning_rate', self.lrn_rate)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)
        optimizer = self._get_optimizer(self.optimizer)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')

        self._extra_train_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
              costs.append(tf.nn.l2_loss(var))
              # tf.summary.histogram(var.op.name, var)
        return tf.multiply(self.weight_decay_rate, tf.add_n(costs))

    def _get_optimizer(self, optimizer_name):
        def adam(): return tf.train.AdamOptimizer(self.lrn_rate)
        def mom():  return tf.train.MomentumOptimizer(self.lrn_rate, 0.9, use_nesterov=True)
        def sgd():  return tf.train.GradientDescentOptimizer(self.lrn_rate)

        optimizer = tf.case(
            {tf.equal(optimizer_name, 'ADAM'): adam(),
             tf.equal(optimizer_name, 'MOM'):  mom()},
            default=sgd(), exclusive=True)
        return optimizer
