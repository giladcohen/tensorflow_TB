from __future__ import division

import lib.logger.logger as logger
from lib.datasets.active_dataset import ActiveDataSet
from lib.datasets.dataset import DataSet
from lib.datasets.dataset_wrapper import DatasetWrapper
from lib.models.resnet_model import ResNet
from lib.models.dml_resnet_model import DMLResNet
from lib.models.wide_resnet_28_10_plus_fc import WideResNet_28_10_plus_fc
from lib.models.wide_resnet_28_10_pool_classes import WideResNet_28_10_pool_classes
from lib.models.wide_resnet_28_10_pool_classes2 import WideResNet_28_10_pool_classes2
from lib.preprocessors.preprocessor import PreProcessor
from lib.trainers.active_learning.all_centers_trainer import AllCentersTrainer
from lib.trainers.active_learning.class_centers_trainer import ClassCentersTrainer
from lib.trainers.active_learning.cross_entropy_trainer import CrossEntropyTrainer
from lib.trainers.active_learning.cross_entropy_trainer2 import CrossEntropyTrainer2
from lib.trainers.active_learning.dynamic_model.kmeans_segments_dynamic_trainer import KMeansSegmentsDynamicTrainer
from lib.trainers.active_learning.dynamic_model.kmeans_segments_knn_dnn_correlation_dynamic_trainer import KMeansSegmentsKnnDnnCorrelationDynamicTrainer
from lib.trainers.active_learning.dynamic_model.knn_dnn_correlation_dynamic_trainer import KnnDnnCorrelationDynamicTrainer
from lib.trainers.active_learning.dynamic_model.most_uncertained_dynamic_trainer import MostUncertainedDynamicTrainer
from lib.trainers.active_learning.dynamic_model.most_uncertained_knn_dynamic_trainer import MostUncertainedKnnDynamicTrainer
from lib.trainers.active_learning.dynamic_model.random_sampler_dynamic_trainer import RandomSamplerDynamicTrainer
from lib.trainers.active_learning.farthest_kmeans_trainer import FarthestKMeansTrainer
from lib.trainers.active_learning.farthest_uncertained_samples_trainer import FarthestUncertainedSamplesTrainer
from lib.trainers.active_learning.kmeans_on_most_uncertained_trainer import KMeansOnMostUncertainedTrainer
from lib.trainers.active_learning.kmeans_segments_balanced_trainer import KMeansSegmentsBalancedTrainer
from lib.trainers.active_learning.kmeans_segments_knn_dnn_correlation_trainer import KMeansSegmentsKnnDnnCorrelationTrainer
from lib.trainers.active_learning.kmeans_segments_most_uncertained_knn import KMeansSegmentsMostUncertainedKNNTrainer
from lib.trainers.active_learning.kmeans_segments_trainer import KMeansSegmentsTrainer
from lib.trainers.active_learning.knn_dnn_correlation_trainer import KnnDnnCorrelationTrainer
from lib.trainers.active_learning.most_uncertained_balanced_trainer import MostUncertainedBalancedTrainer
from lib.trainers.active_learning.most_uncertained_knn_trainer import MostUncertainedKnnTrainer
from lib.trainers.active_learning.most_uncertained_trainer import MostUncertainedTrainer
from lib.trainers.active_learning.random_sampler_trainer import RandomSamplerTrainer
from lib.trainers.classification_trainer import ClassificationTrainer
from lib.trainers.dml_classification_trainer import DMLClassificationTrainer
from lib.trainers.hooks.decay_by_score_setter import DecayByScoreSetter
from lib.trainers.hooks.fixed_schedule_setter import FixedScheduleSetter
from lib.trainers.hooks.learning_rate_setter_base import LearningRateSetterBase
from lib.testers.knn_classifier_tester import KNNClassifierTester


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
        self.tester               = self.prm.test.test_control.TESTER

    def get_dataset(self, preprocessor):
        available_datasets = {'cifar10'         : DataSet,
                              'cifar100'        : DataSet,
                              'active_cifar10'  : ActiveDataSet,
                              'active_cifar100' : ActiveDataSet}
        if self.dataset_name in available_datasets:
            dataset = DatasetWrapper(self.dataset_name + '_wrapper', self.prm)

            train_dataset = available_datasets[self.dataset_name](self.dataset_name + '_train', self.prm, preprocessor)
            train_dataset.initialize_pool()
            dataset.set_train_dataset(train_dataset)
            self.log.info('get_train_dataset: returning ' + str(train_dataset))

            validation_dataset = available_datasets[self.dataset_name](self.dataset_name + '_validation', self.prm, preprocessor)
            dataset.set_validation_dataset(validation_dataset)
            self.log.info('get_validation_dataset: returning ' + str(validation_dataset))

            test_dataset = available_datasets[self.dataset_name](self.dataset_name + '_test', self.prm, preprocessor)
            dataset.set_test_dataset(test_dataset)
            self.log.info('get_test_dataset: returning ' + str(test_dataset))

            dataset.split_train_validation()

            return dataset
        else:
            err_str = 'get_dataset: dataset {} was not found. Available datasets are: {}'.format(self.dataset_name, available_datasets.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_model(self):
        available_networks = {'Wide-Resnet-28-10'               : ResNet,
                              'DML-Wide-Resnet-28-10'           : DMLResNet,
                              'Wide-Resnet-28-10_plus_fc'       : WideResNet_28_10_plus_fc,
                              'Wide-Resnet-28-10_pool_classes'  : WideResNet_28_10_pool_classes,
                              'Wide-Resnet-28-10_pool_classes2' : WideResNet_28_10_pool_classes2}
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
        available_trainers = {'simple'                               : ClassificationTrainer,
                              'dml'                                  : DMLClassificationTrainer,
                              'random_sampler'                       : RandomSamplerTrainer,
                              'all_centers'                          : AllCentersTrainer,
                              'class_centers'                        : ClassCentersTrainer,
                              'most_uncertained'                     : MostUncertainedTrainer,
                              'most_uncertained_balanced'            : MostUncertainedBalancedTrainer,
                              'cross_entropy'                        : CrossEntropyTrainer,
                              'cross_entropy2'                       : CrossEntropyTrainer2,
                              'kmeans_on_most_uncertained'           : KMeansOnMostUncertainedTrainer,
                              'knn_dnn_correlation'                  : KnnDnnCorrelationTrainer,
                              'most_uncertained_knn'                 : MostUncertainedKnnTrainer,
                              'farthest_uncertained_samples'         : FarthestUncertainedSamplesTrainer,
                              'kmeans_segments'                      : KMeansSegmentsTrainer,
                              'kmeans_segments_balanced'             : KMeansSegmentsBalancedTrainer,
                              'farthest_kmeans'                      : FarthestKMeansTrainer,
                              'kmeans_segments_most_uncertained_knn' : KMeansSegmentsMostUncertainedKNNTrainer,
                              'kmeans_segments_dynamic'              : KMeansSegmentsDynamicTrainer,
                              'kmeans_segments_knn_dnn_correlation'  : KMeansSegmentsKnnDnnCorrelationTrainer,
                              'knn_dnn_correlation_dynamic'          : KnnDnnCorrelationDynamicTrainer,
                              'random_sampler_dynamic'               : RandomSamplerDynamicTrainer,
                              'most_uncertained_dynamic'             : MostUncertainedDynamicTrainer,
                              'kmeans_segments_knn_dnn_correlation_dynamic' : KMeansSegmentsKnnDnnCorrelationDynamicTrainer,
                              'most_uncertained_knn_dynamic'         : MostUncertainedKnnDynamicTrainer}
        if self.trainer in available_trainers:
            trainer = available_trainers[self.trainer](self.trainer, self.prm, model, dataset)
            self.log.info('get_trainer: returning ' + str(trainer))
            trainer.build()
            return trainer
        else:
            err_str = 'get_trainer: trainer {} was not found. Available trainers are: {}'.format(self.trainer, available_trainers.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_tester(self, model, dataset):
        available_testers = {'knn_classifier' : KNNClassifierTester}
        if self.tester in available_testers:
            tester = available_testers[self.tester](self.tester, self.prm, model, dataset)
            self.log.info('get_tester: returning ' + str(tester))
            tester.build()
            return tester
        else:
            err_str = 'get_tester: tester {} was not found. Available testers are: {}'.format(self.tester, available_testers.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_learning_rate_setter(self, model, trainset_dataset, retention):
        available_setters = {'fixed': LearningRateSetterBase,
                             'fixed_schedule': FixedScheduleSetter,
                             'decay_by_score': DecayByScoreSetter}
        if self.learning_rate_setter in available_setters:
            setter = available_setters[self.learning_rate_setter](self.learning_rate_setter, self.prm, model, trainset_dataset, retention)
            self.log.info('get_learning_rate_setter: returning ' + str(setter))
            return setter
        else:
            err_str = 'get_learning_rate_setter: learning_rate_setter {} was not found. Available setters are: {}'.format(self.learning_rate_setter, available_setters.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)
