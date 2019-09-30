import argparse
import os
import sys
import warnings

cwd = os.getcwd() # tensorflow-TB
sys.path.insert(0, cwd)

from tensorflow_TB.lib.logger.logging_config import logging_config
from tensorflow_TB.utils.parameters import Parameters
from tensorflow_TB.utils.factories import Factories
import tensorflow as tf
from tensorflow_TB.utils.misc import query_yes_no

def get_params(train_config):
    """get params and save them to root dir"""
    prm = Parameters()
    prm.override(train_config)
    parameter_file = os.path.join(prm.train.train_control.ROOT_DIR, 'parameters.ini')
    log_file       = os.path.join(prm.train.train_control.ROOT_DIR, 'minirunt.log')

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
    parser.add_argument('-c', help='Train configuration file', action='store')
    args = parser.parse_args()

    train_config = args.c
    if not os.path.isfile(train_config):
        raise AssertionError('Can not find file: {}'.format(train_config))

    prm = get_params(train_config)

    dev = prm.network.DEVICE
    with tf.device(dev):
        train(prm)
