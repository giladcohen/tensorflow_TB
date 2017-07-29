from __future__ import division

import lib.logger.logger as logger
from lib.preprocessors.preprocessor import PreProcessor
from lib.trainers.classification_trainer import ClassificationTrainer
from lib.trainers.active_trainer import ActiveTrainer
from lib.trainers.hooks.learning_rate_setter_base import LearningRateSetterBase
from lib.trainers.hooks.fixed_schedule_setter import FixedScheduleSetter
from lib.trainers.hooks.precision_decay_setter import PrecisionDecaySetter
from lib.models.resnet_model import ResNet
from lib.datasets.dataset import DataSet
from lib.datasets.passive_dataset import PassiveDataSet
from lib.datasets.active_dataset import ActiveDataSet


class Factories(object):
    """Class which encapsulate all factories in the TB"""

    def __init__(self, prm):
        self.log = logger.get_logger('factories')
        self.prm = prm

        self.dataset_name         = self.prm.dataset.DATASET_NAME
        self.preprocessor         = self.prm.network.pre_processing.PREPROCESSOR
        self.trainer              = self.prm.train.train_control.TRAINER
        self.architecture         = self.prm.network.ARCHITECTURE
        self.learning_rate_setter = self.prm.train.train_control.learning_rate_setter.LEARNING_RATE_SETTER

    def get_train_dataset(self, preprocessor):
        available_datasets = {'cifar10': DataSet, 'passive_cifar10': PassiveDataSet, 'active_cifar10': ActiveDataSet}
        if self.dataset_name in available_datasets:
            dataset = available_datasets[self.dataset_name](self.dataset_name + '_train', self.prm, preprocessor)
            self.log.info('get_train_dataset: returning ' + str(dataset))
            return dataset
        else:
            err_str = 'get_train_dataset: dataset {} was not found. Available datasets are: {}'.format(self.dataset_name, available_datasets.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_validation_dataset(self, preprocessor):
        available_datasets = {'cifar10': DataSet, 'passive_cifar10': DataSet, 'active_cifar10': DataSet}
        if self.dataset_name in available_datasets:
            dataset = available_datasets[self.dataset_name](self.dataset_name + '_validation', self.prm, preprocessor)
            self.log.info('get_validation_dataset: returning ' + str(dataset))
            return dataset
        else:
            err_str = 'get_validation_dataset: dataset {} was not found. Available datasets are: {}'.format(self.dataset_name, available_datasets.keys())
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
        available_trainers = {'simple': ClassificationTrainer, 'active': ActiveTrainer}
        if self.trainer in available_trainers:
            trainer = available_trainers[self.trainer](self.trainer, self.prm, model, dataset)
            self.log.info('get_trainer: returning ' + str(trainer))
            return trainer
        else:
            err_str = 'get_trainer: trainer {} was not found. Available trainers are: {}'.format(self.trainer, available_trainers.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_learning_rate_setter(self, model, trainset_dataset, precision_retention):
        available_setters = {'fixed': LearningRateSetterBase, 'fixed_schedule': FixedScheduleSetter,
                             'precision_decay': PrecisionDecaySetter}
        if self.learning_rate_setter in available_setters:
            setter = available_setters[self.learning_rate_setter](self.learning_rate_setter, self.prm, model, trainset_dataset, precision_retention)
            self.log.info('get_learning_rate_setter: returning ' + str(setter))
            return setter
        else:
            err_str = 'get_learning_rate_setter: learning_rate_setter {} was not found. Available setters are: {}'.format(self.learning_rate_setter, available_setters.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)
