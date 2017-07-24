import argparse
import os
import sys

cwd = os.getcwd() # Resnet_KNN
sys.path.insert(0, cwd)

import lib.logger.logger as logger
from lib.logger.logging_config import logging_config
from utils.parameters import Parameters
from utils.utils import Factories

logging = logging_config()
logging.disable(logging.DEBUG)
log = logger.get_logger('main')

def train(train_config):
    prm = Parameters()
    prm.override(train_config)
    factories = Factories(prm)

    model = factories.get_model()
    trainer = factories.get_trainer()
    print('hey')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', help='Train configuration file', action='store')
    args = parser.parse_args()

    train_config = args.c
    if not os.path.isfile(train_config):
        log.error('Can not find file: {}'.format(train_config))
        exit(-1)

    train(train_config)
