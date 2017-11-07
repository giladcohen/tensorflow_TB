from __future__ import division

import lib.logger.logger as logger
from lib.preprocessors.preprocessor import PreProcessor
from lib.trainers.classification_trainer import ClassificationTrainer
from lib.trainers.random_sampler_trainer import RandomSamplerTrainer
from lib.trainers.all_centers_trainer import AllCentersTrainer
from lib.trainers.class_centers_trainer import ClassCentersTrainer
from lib.trainers.most_uncertained_trainer import MostUncertainedTrainer
from lib.trainers.most_uncertained_balanced_trainer import MostUncertainedBalancedTrainer
from lib.trainers.cross_entropy_trainer import CrossEntropyTrainer
from lib.trainers.hooks.learning_rate_setter_base import LearningRateSetterBase
from lib.trainers.hooks.fixed_schedule_setter import FixedScheduleSetter
from lib.trainers.hooks.decay_by_score_setter import DecayByScoreSetter
from lib.models.wide_resnet_28_10 import WideResNet_28_10
from lib.models.wide_resnet_28_10_plus_fc import WideResNet_28_10_plus_fc
from lib.datasets.dataset_wrapper import DatasetWrapper
from lib.datasets.dataset import DataSet
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

    def get_dataset(self, preprocessor):
        available_datasets = {'cifar10': DataSet, 'active_cifar10': ActiveDataSet}
        if self.dataset_name in available_datasets:
            dataset = DatasetWrapper(self.dataset_name + '_wrapper', self.prm)

            train_dataset = available_datasets[self.dataset_name](self.dataset_name + '_train', self.prm, preprocessor)
            train_dataset.initialize_pool()
            dataset.set_train_dataset(train_dataset)
            self.log.info('get_train_dataset: returning ' + str(train_dataset))

            validation_dataset = available_datasets[self.dataset_name](self.dataset_name + '_validation', self.prm, preprocessor)
            validation_dataset.initialize_pool()
            dataset.set_validation_dataset(validation_dataset)
            self.log.info('get_validation_dataset: returning ' + str(validation_dataset))

            return dataset
        else:
            err_str = 'get_dataset: dataset {} was not found. Available datasets are: {}'.format(self.dataset_name, available_datasets.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_model(self):
        available_networks = {'Wide-Resnet-28-10': WideResNet_28_10, 'Wide-Resnet-28-10_plus_fc': WideResNet_28_10_plus_fc}
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
        available_trainers = {'simple'                    : ClassificationTrainer,
                              'random_sampler'            : RandomSamplerTrainer,
                              'all_centers'               : AllCentersTrainer,
                              'class_centers'             : ClassCentersTrainer,
                              'most_uncertained'          : MostUncertainedTrainer,
                              'most_uncertained_balanced' : MostUncertainedBalancedTrainer,
                              'cross_entropy'             : CrossEntropyTrainer}
        if self.trainer in available_trainers:
            trainer = available_trainers[self.trainer](self.trainer, self.prm, model, dataset)
            self.log.info('get_trainer: returning ' + str(trainer))
            trainer.build()
            return trainer
        else:
            err_str = 'get_trainer: trainer {} was not found. Available trainers are: {}'.format(self.trainer, available_trainers.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_learning_rate_setter(self, model, trainset_dataset, retention):
        available_setters = {'fixed': LearningRateSetterBase, 'fixed_schedule': FixedScheduleSetter,
                             'decay_by_score': DecayByScoreSetter}
        if self.learning_rate_setter in available_setters:
            setter = available_setters[self.learning_rate_setter](self.learning_rate_setter, self.prm, model, trainset_dataset, retention)
            self.log.info('get_learning_rate_setter: returning ' + str(setter))
            return setter
        else:
            err_str = 'get_learning_rate_setter: learning_rate_setter {} was not found. Available setters are: {}'.format(self.learning_rate_setter, available_setters.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)
