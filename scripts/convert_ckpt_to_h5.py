from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf
import os
import sys

from threading import Thread

if sys.version_info[0] < 3:
    from Queue import Queue
else:
    import queue as Queue

import darkon.darkon as darkon

from cleverhans.attacks import FastGradientMethod, DeepFool
from tensorflow.python.platform import flags
from cleverhans.loss import CrossEntropy, WeightDecay, WeightedSum
from tensorflow_TB.lib.models.darkon_replica_model import DarkonReplica
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval
from tensorflow_TB.utils.misc import one_hot
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tensorflow_TB.lib.datasets.influence_feeder_val_test import MyFeederValTest
from tensorflow_TB.utils.misc import np_evaluate
import copy
import pickle
import imageio

FLAGS = flags.FLAGS


flags.DEFINE_float('weight_decay', 0.0004, 'weight decay')
flags.DEFINE_string('checkpoint_name', 'cifar100/log_300419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000_ls_0.01', 'checkpoint name')
flags.DEFINE_string('dataset', 'cifar100', 'datasset: cifar10/100 or svhn')
