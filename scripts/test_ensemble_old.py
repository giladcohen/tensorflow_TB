import argparse
import os
import sys
import warnings

cwd = os.getcwd() # tensorflow-TB
sys.path.insert(0, cwd)

from lib.logger.logging_config import logging_config
from utils.parameters import Parameters
from utils.factories import Factories
import tensorflow as tf
from utils.misc import query_yes_no

def get_params(test_config):
    """get params and save them to root dir"""
    prm = Parameters()

    # get giles paths
    prm.override(test_config)
    test_parameter_file = os.path.join(prm.train.train_control.ROOT_DIR, 'test_parameters.ini')
    log_file            = os.path.join(prm.train.train_control.ROOT_DIR, 'test.log')

    ret = True
    if os.path.isfile(test_parameter_file):
        warnings.warn('Test parameter file {} already exists'.format(test_parameter_file))
        ret = query_yes_no('Overwrite parameter file?')

    if ret:
        dir = os.path.dirname(test_parameter_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        prm.save(test_parameter_file)

    logging = logging_config(log_file)
    logging.disable(logging.DEBUG)

    return prm

def test(prm):
    tf.set_random_seed(prm.SUPERSEED)
    factories = Factories(prm)

    tester      = factories.get_tester(model=None, dataset=None)
    tester.print_stats() #debug

    tester.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', help='Test configuration file', action='store')
    args = parser.parse_args()

    test_config = args.c
    if not os.path.isfile(test_config):
        raise AssertionError('Can not find file: {}'.format(test_config))

    prm = get_params(test_config)

    dev = prm.network.DEVICE
    with tf.device(dev):
        test(prm)
