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

def get_params(train_config, parser_args=None):
    """get params and save them to root dir"""
    prm = Parameters()
    prm.override(train_config)

    # get manual train parameters from config:
    if parser_args is not None:
        # overriding some parameters manually from parser:
        prm.train.train_control.ROOT_DIR           = parser_args.ROOT_DIR
        prm.train.train_control.TRAIN_DIR          = parser_args.ROOT_DIR + '/train'
        prm.train.train_control.EVAL_DIR           = parser_args.ROOT_DIR + '/validation'
        prm.train.train_control.TEST_DIR           = parser_args.ROOT_DIR + '/test'
        prm.train.train_control.PREDICTION_DIR     = parser_args.ROOT_DIR + '/prediction'
        prm.train.train_control.CHECKPOINT_DIR     = parser_args.ROOT_DIR + '/checkpoint'
        prm.SUPERSEED                              = int(parser_args.SUPERSEED)
        prm.network.MULTI_SF                       = (parser_args.MULTI_SF == 'True')
        prm.network.ARCHITECTURE                   = parser_args.ARCHITECTURE
        prm.dataset.TRAIN_SET_SIZE                 = int(parser_args.TRAIN_SET_SIZE)
        prm.network.system.DROPOUT_KEEP_PROB       = float(parser_args.DROPOUT_KEEP_PROB)

    ROOT_DIR = prm.train.train_control.ROOT_DIR

    parameter_file = os.path.join(ROOT_DIR, 'parameters.ini')
    log_file       = os.path.join(ROOT_DIR, 'minirunt.log')

    ret = True
    if os.path.isfile(parameter_file):
        warnings.warn('Parameter file {} already exists'.format(parameter_file))
        ret = query_yes_no('Overwrite parameter file?')

    if ret:
        dir = os.path.dirname(parameter_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        prm.save(parameter_file)

    logging = logging_config(log_file)
    logging.disable(logging.DEBUG)

    return prm

def train(prm):
    tf.set_random_seed(prm.SUPERSEED)
    factories = Factories(prm)

    model        = factories.get_model()
    model.print_stats() #debug

    dataset = factories.get_dataset()
    dataset.print_stats() #debug

    trainer      = factories.get_trainer(model, dataset)
    trainer.print_stats() #debug

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # for automating KNN testing
    parser.add_argument('--ROOT_DIR'         , action='store')
    parser.add_argument('--SUPERSEED'        , action='store')
    parser.add_argument('--MULTI_SF'         , action='store')
    parser.add_argument('--ARCHITECTURE'     , action='store')
    parser.add_argument('--TRAIN_SET_SIZE'   , action='store')
    parser.add_argument('--DROPOUT_KEEP_PROB', action='store')

    parser.add_argument('-c', help='Train configuration file', action='store')
    args = parser.parse_args()

    train_config = args.c
    if not os.path.isfile(train_config):
        raise AssertionError('Can not find file: {}'.format(train_config))

    prm = get_params(train_config, parser_args=args)

    dev = prm.network.DEVICE
    with tf.device(dev):
        train(prm)
