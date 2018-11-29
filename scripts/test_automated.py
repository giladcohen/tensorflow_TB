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
from utils.misc import query_yes_no, get_timestamp



def get_params(test_config, parser_args=None):
    """get params and save them to root dir"""

    # Just to get the ROOT_DIR and save prm test_config
    prm = Parameters()
    prm.override(test_config)

    # get manual test parameters from config:
    if parser_args is not None:
        # overriding some parameters manually from parser:
        prm.train.train_control.ROOT_DIR           = parser_args.ROOT_DIR
        prm.train.train_control.TEST_DIR           = parser_args.ROOT_DIR + '/test'
        prm.train.train_control.PREDICTION_DIR     = parser_args.ROOT_DIR + '/prediction'
        prm.train.train_control.CHECKPOINT_DIR     = parser_args.ROOT_DIR + '/checkpoint'
        prm.test.test_control.KNN_WEIGHTS          = parser_args.KNN_WEIGHTS
        prm.test.test_control.KNN_NORM             = parser_args.KNN_NORM
        prm.train.train_control.PCA_REDUCTION      = (parser_args.PCA_REDUCTION == 'True')
        prm.train.train_control.PCA_EMBEDDING_DIMS = int(parser_args.PCA_EMBEDDING_DIMS)
        prm.test.test_control.KNN_NEIGHBORS        = int(parser_args.KNN_NEIGHBORS)
        prm.test.test_control.DUMP_NET             = (parser_args.DUMP_NET == 'True')
        prm.test.test_control.LOAD_FROM_DISK       = (parser_args.LOAD_FROM_DISK == 'True')
        prm.test.test_control.CHECKPOINT_FILE      = parser_args.CHECKPOINT_FILE

    ROOT_DIR = prm.train.train_control.ROOT_DIR

    # get time stamp
    ts = get_timestamp()

    # get files paths
    parameter_file      = os.path.join(ROOT_DIR, 'parameters.ini')
    test_parameter_file = os.path.join(ROOT_DIR, 'test_parameters_'+ts+'.ini')
    all_parameter_file  = os.path.join(ROOT_DIR, 'all_parameters_'+ts+'.ini')
    log_file            = os.path.join(ROOT_DIR, 'test_'+ts+'.log')
    logging = logging_config(log_file)
    logging.disable(logging.DEBUG)

    if not os.path.isfile(parameter_file):
        raise AssertionError('Can not find file: {}'.format(parameter_file))

    dir = os.path.dirname(test_parameter_file)
    if not os.path.exists(dir):
        os.makedirs(dir)
    prm.save(test_parameter_file)

    # Done saving test parameters. Now doing the integration:
    prm = Parameters()
    prm.override(parameter_file)
    prm.override(test_parameter_file)
    if parser_args is not None:
        # overriding some parameters manually from parser:
        prm.train.train_control.ROOT_DIR           = parser_args.ROOT_DIR
        prm.train.train_control.TEST_DIR           = parser_args.ROOT_DIR + '/test'
        prm.train.train_control.PREDICTION_DIR     = parser_args.ROOT_DIR + '/prediction'
        prm.train.train_control.CHECKPOINT_DIR     = parser_args.ROOT_DIR + '/checkpoint'
        prm.test.test_control.KNN_WEIGHTS          = parser_args.KNN_WEIGHTS
        prm.test.test_control.KNN_NORM             = parser_args.KNN_NORM
        prm.train.train_control.PCA_REDUCTION      = (parser_args.PCA_REDUCTION == 'True')
        prm.train.train_control.PCA_EMBEDDING_DIMS = int(parser_args.PCA_EMBEDDING_DIMS)
        prm.test.test_control.KNN_NEIGHBORS        = int(parser_args.KNN_NEIGHBORS)
        prm.test.test_control.DUMP_NET             = (parser_args.DUMP_NET == 'True')
        prm.test.test_control.LOAD_FROM_DISK       = (parser_args.LOAD_FROM_DISK == 'True')
        prm.test.test_control.CHECKPOINT_FILE      = parser_args.CHECKPOINT_FILE

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

    # for automating KNN testing
    parser.add_argument('--ROOT_DIR'          , action='store')
    parser.add_argument('--KNN_WEIGHTS'       , action='store')
    parser.add_argument('--KNN_NORM'          , action='store')
    parser.add_argument('--PCA_REDUCTION'     , action='store')
    parser.add_argument('--PCA_EMBEDDING_DIMS', action='store')
    parser.add_argument('--KNN_NEIGHBORS'     , action='store')
    parser.add_argument('--DUMP_NET'          , action='store')
    parser.add_argument('--LOAD_FROM_DISK'    , action='store')
    parser.add_argument('--CHECKPOINT_FILE'   , action='store')

    parser.add_argument('-c', help='Test configuration file', action='store')
    args = parser.parse_args()

    test_config = args.c
    if not os.path.isfile(test_config):
        raise AssertionError('Can not find file: {}'.format(test_config))

    prm = get_params(test_config, parser_args=args)

    dev = prm.network.DEVICE
    with tf.device(dev):
        test(prm)
