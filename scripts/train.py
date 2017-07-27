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

logging = logging_config()
logging.disable(logging.DEBUG)
log = logger.get_logger('main')

def train(prm):
    factories = Factories(prm)

    model        = factories.get_model()
    model.print_stats() #debug

    preprocessor = factories.get_preprocessor()
    preprocessor.print_stats() #debug

    train_dataset      = factories.get_train_dataset(preprocessor)
    validation_dataset = factories.get_validation_dataset(preprocessor)

    dataset_wrapper =  DatasetWrapper(prm.dataset.DATASET_NAME + '_wrapper', prm, train_dataset, validation_dataset)
    dataset_wrapper.print_stats()

    trainer      = factories.get_trainer(model, dataset_wrapper)
    trainer.print_stats() #debug

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', help='Train configuration file', action='store')
    args = parser.parse_args()

    train_config = args.c
    if not os.path.isfile(train_config):
        log.error('Can not find file: {}'.format(train_config))
        exit(-1)

    prm = Parameters()
    prm.override(train_config)
    dev = prm.network.DEVICE

    with tf.device(dev):
        train(prm)
