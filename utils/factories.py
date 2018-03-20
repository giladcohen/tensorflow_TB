from __future__ import division

import lib.logger.logger as logger
from lib.datasets.dataset_wrapper import DatasetWrapper
from lib.datasets.active_dataset_wrapper import ActiveDatasetWrapper
from lib.datasets.semi_supervised_dataset_wrapper import SemiSupervisedDatasetWrapper
from lib.datasets.nexet_dataset_wrapper import NexetDatasetWrapper
from lib.models.resnet_model import ResNet
from lib.models.dml_resnet_model import DMLResNet
from lib.models.ssd_mobilenet import SSDMobileNet
from lib.models.wide_resnet_28_10_plus_fc import WideResNet_28_10_plus_fc
from lib.models.wide_resnet_28_10_pool_classes import WideResNet_28_10_pool_classes
from lib.models.wide_resnet_28_10_pool_classes2 import WideResNet_28_10_pool_classes2
from lib.models.wide_resnet_28_10_wo_last_relu import WideResNet_28_10_wo_last_relu
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
from lib.trainers.active_learning.random_sampler_trainer_qad import RandomSamplerTrainerQAD
from lib.trainers.active_learning.random_sampler_trainer import RandomSamplerTrainer
from lib.trainers.classification_trainer import ClassificationTrainer
from lib.trainers.dml_classification_trainer import DMLClassificationTrainer
from lib.trainers.active_trainer import ActiveTrainer
from lib.trainers.semi_supervised_trainer import SemiSupervisedTrainer
from lib.trainers.object_detection_trainer import ObjectDetectionTrainer
from lib.trainers.hooks.decay_by_score_setter import DecayByScoreSetter
from lib.trainers.hooks.fixed_schedule_setter import FixedScheduleSetter
from lib.trainers.hooks.learning_rate_setter_base import LearningRateSetterBase
from lib.testers.knn_classifier_tester import KNNClassifierTester
from lib.testers.ensemble_tester import EnsembleTester
import lib.trainers.active_learning.active_learning_select_functions as alf


class Factories(object):
    """Class which encapsulate all factories in the TB"""

    def __init__(self, prm):
        self.log = logger.get_logger('factories')
        self.prm = prm

        self.dataset_name               = self.prm.dataset.DATASET_NAME
        self.trainer                    = self.prm.train.train_control.TRAINER
        self.architecture               = self.prm.network.ARCHITECTURE
        self.learning_rate_setter       = self.prm.train.train_control.learning_rate_setter.LEARNING_RATE_SETTER
        self.tester                     = self.prm.test.test_control.TESTER
        self.active_selection_criterion = self.prm.train.train_control.ACTIVE_SELECTION_CRITERION

    def get_dataset(self):
        available_datasets = {'cifar10'         : DatasetWrapper,
                              'cifar100'        : DatasetWrapper,
                              'active_cifar10'  : ActiveDatasetWrapper,
                              'active_cifar100' : ActiveDatasetWrapper,
                              'semi_cifar10'    : SemiSupervisedDatasetWrapper,
                              'semi_cifar100'   : SemiSupervisedDatasetWrapper,
                              'nexet'           : NexetDatasetWrapper}
        if self.dataset_name in available_datasets:
            dataset = available_datasets[self.dataset_name](self.dataset_name + '_dataset_wrapper', self.prm)
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
                              'Wide-Resnet-28-10_pool_classes2' : WideResNet_28_10_pool_classes2,
                              'Wide-Resnet-28-10_wo_last_relu'  : WideResNet_28_10_wo_last_relu,
                              'SSD-MobileNet'                   : SSDMobileNet}
        if self.architecture in available_networks:
            model = available_networks[self.architecture](self.architecture, self.prm)
            self.log.info('get_model: returning ' + str(model))
            return model
        else:
            err_str = 'get_model: model {} was not found. Available models are: {}'.format(self.architecture, available_networks.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_trainer(self, model, dataset):
        available_trainers = {'simple'                               : ClassificationTrainer,
                              'dml'                                  : DMLClassificationTrainer,
                              'active'                               : ActiveTrainer,
                              'semi_supervised'                      : SemiSupervisedTrainer,
                              'object_detection'                     : ObjectDetectionTrainer,
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
                              'most_uncertained_knn_dynamic'         : MostUncertainedKnnDynamicTrainer,
                              'random_sampler_trainer_qad'           : RandomSamplerTrainerQAD}
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
        available_testers = {'knn_classifier'     : KNNClassifierTester,
                             'ensemble_classifier': EnsembleTester}
        if self.tester in available_testers:
            tester = available_testers[self.tester](self.tester, self.prm, model, dataset)
            self.log.info('get_tester: returning ' + str(tester))
            tester.build()
            return tester
        else:
            err_str = 'get_tester: tester {} was not found. Available testers are: {}'.format(self.tester, available_testers.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_learning_rate_setter(self, model, retention):
        available_setters = {'fixed': LearningRateSetterBase,
                             'fixed_schedule': FixedScheduleSetter,
                             'decay_by_score': DecayByScoreSetter}
        if self.learning_rate_setter in available_setters:
            setter = available_setters[self.learning_rate_setter](self.learning_rate_setter, self.prm, model, retention)
            self.log.info('get_learning_rate_setter: returning ' + str(setter))
            return setter
        else:
            err_str = 'get_learning_rate_setter: learning_rate_setter {} was not found. Available setters are: {}'.format(self.learning_rate_setter, available_setters.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_active_selection_fn(self):
        available_functions = {'min_mul_dnn_max_knn_same'           : alf.min_mul_dnn_max_knn_same,
                               'most_uncertained_following_min_corr': alf.most_uncertained_following_min_corr,
                               'min_corr_following_most_uncertained': alf.min_corr_following_most_uncertained}
        if self.active_selection_criterion in available_functions:
            function = available_functions[self.active_selection_criterion]
            self.log.info('get_active_selection_fn: returning ' + self.active_selection_criterion)
            return function
        else:
            err_str = 'get_active_selection_fn: active_selection_criterion {} was not found. Available functions are: {}'.format(self.active_selection_criterion, available_functions.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

