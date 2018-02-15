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
    prm.override(test_config)  # just to get the LOG_DIR_LIST[0]
    train_log_dir       = prm.test.ensemble.LOG_DIR_LIST[0]

    parameter_file      = os.path.join(train_log_dir, 'parameters.ini')
    test_parameter_file = os.path.join(prm.train.train_control.ROOT_DIR, 'test_parameters.ini')
    all_parameter_file  = os.path.join(prm.train.train_control.ROOT_DIR, 'all_parameters.ini')
    log_file            = os.path.join(prm.train.train_control.ROOT_DIR, 'test.log')

    if not os.path.isfile(parameter_file):
        raise AssertionError('Can not find file: {}'.format(parameter_file))

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

    # Done saving test parameters. Now doing the integration:
    prm = Parameters()
    prm.override(parameter_file)
    prm.override(test_parameter_file)

    ret = True
    if os.path.isfile(all_parameter_file):
        warnings.warn('All parameter file {} already exists'.format(all_parameter_file))
        ret = query_yes_no('Overwrite parameter file?')

    if ret:
        dir = os.path.dirname(all_parameter_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        prm.save(all_parameter_file)

    return prm

def test(prm):
    tf.set_random_seed(prm.SUPERSEED)
    factories = Factories(prm)

    model        = factories.get_model()
    model.print_stats() #debug

    dataset = factories.get_dataset()
    dataset.print_stats() #debug

    tester      = factories.get_tester(model, dataset)
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
