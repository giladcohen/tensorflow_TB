import argparse
import os
import sys

cwd = os.getcwd() # tensorflow-TB
sys.path.insert(0, cwd)

import lib.logger.logger as logger
from lib.logger.logging_config import logging_config
from utils.parameters import Parameters
from utils.factories import Factories
from lib.datasets.dataset_wrapper import DatasetWrapper
import tensorflow as tf
from utils.misc import query_yes_no

def get_params(train_config):
    """get params and save them to root dir"""
    prm = Parameters()
    prm.override(train_config)
    parameter_file = os.path.join(prm.train.train_control.ROOT_DIR, 'parameters.ini')
    log_file       = os.path.join(prm.train.train_control.ROOT_DIR, 'minirunt.log')

    logging = logging_config(log_file)
    logging.disable(logging.DEBUG)
    log = logger.get_logger('main')

    ret = True
    if os.path.isfile(parameter_file):
        log.warning('Parameter file {} already exists'.format(parameter_file))
        ret = query_yes_no('Overwrite parameter file?')

    if ret:
        dir = os.path.dirname(parameter_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        prm.save(parameter_file)

    return prm

def train(prm):
    tf.set_random_seed(prm.SUPERSEED)
    factories = Factories(prm)

    model        = factories.get_model()
    model.print_stats() #debug

    preprocessor = factories.get_preprocessor()
    preprocessor.print_stats() #debug

    dataset = factories.get_dataset(preprocessor)
    dataset.print_stats() #debug

    trainer      = factories.get_trainer(model, dataset)
    trainer.print_stats() #debug

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', help='Train configuration file', action='store')
    args = parser.parse_args()

    train_config = args.c
    if not os.path.isfile(train_config):
        raise AssertionError('Can not find file: {}'.format(train_config))

    prm = get_params(train_config)

    dev = prm.network.DEVICE
    with tf.device(dev):
        train(prm)
