from __future__ import division

import lib.logger.logger as logger
from lib.preprocessors.preprocessor import PreProcessor
from lib.trainers.passive_trainer import PassiveTrainer
from lib.models.resnet_model import ResNet
from lib.datasets.dataset_wrapper import DatasetWrapper


class Factories(object):
    """Class which encapsulate all factories in the TB"""

    def __init__(self, prm):
        self.log = logger.get_logger('factories')
        self.prm = prm

        self.dataset = self.prm.dataset.DATASET_NAME
        self.preprocessor = self.prm.network.pre_processing.PREPROCESSOR
        self.trainer = prm.train.train_control.TRAINER
        self.architecture = prm.network.ARCHITECTURE

    def get_dataset(self, preprocessor):
        available_datasets = {'cifar10': DatasetWrapper, 'cifar100': DatasetWrapper}
        if self.dataset in available_datasets:
            dataset = available_datasets[self.dataset](self.dataset, self.prm, preprocessor)
            self.log.info('get_dataset: returning ' + str(dataset))
            return dataset
        else:
            err_str = 'get_dataset: dataset {} was not found. Available datasets are: {}'.format(self.dataset, available_datasets.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_model(self):
        available_networks = {'Wide-Resnet-28-10': ResNet}
        if self.architecture in available_networks:
            model = available_networks[self.architecture](self.architecture, self.prm)
            self.log.info('get_model: returning ' + str(model))
            return model
        else:
            err_str = 'get_model: model {} was not found. Available models are: {}'.format(self.architecture, available_networks.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_preprocessor(self):
        available_preprocessors = {'preprocessor_drift_flip': PreProcessor}
        if self.preprocessor in available_preprocessors:
            preprocessor = available_preprocessors[self.preprocessor](self.preprocessor, self.prm)
            self.log.info('get_preprocessor: returning ' + str(preprocessor))
            return preprocessor
        else:
            err_str = 'get_preprocessor: preprocessor {} was not found. Available preprocessors are: {}'.format(self.preprocessor, available_preprocessors.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_trainer(self, model, dataset):
        available_trainers = {'passive': PassiveTrainer}
        if self.trainer in available_trainers:
            trainer = available_trainers[self.trainer](self.trainer, self.prm, model, dataset)
            self.log.info('get_trainer: returning ' + str(trainer))
            return trainer
        else:
            err_str = 'get_trainer: trainer {} was not found. Available trainers are: {}'.format(self.trainer, available_trainers.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)
