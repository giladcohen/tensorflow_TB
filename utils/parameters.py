from __future__ import absolute_import

import os
import configparser
import utils.ini_parsers as parser_utils
import numpy as np


class Parameters(parser_utils.FrozenClass):
    def __init__(self):
        super(Parameters, self).__init__()

        self.DEBUG_MODE = None                   # boolean
        self.SUPERSEED = None
        self.network = ParametersNetwork()
        self.dataset = ParametersDataset()
        self.train   = ParametersTrain()
        self.test    = ParametersTest()

        self._freeze()  # no new attributes after this point.

    def save(self, out_file, do_save_none=True):
        """
        Save  parameters. Undefined parameters (with None value) are saved with value = None
        :param out_file: Path to file to be saved
        """
        self.log.info('Configuration saved to file: {}'.format(out_file))

        # add the settings to the structure of the file, and lets write it out...
        config = configparser.ConfigParser()
        config._interpolation = configparser.ExtendedInterpolation()

        config.optionxform = str

        root_section = 'self'
        config.add_section(root_section)
        self.set_to_config(do_save_none, root_section, config, 'DEBUG_MODE', self.DEBUG_MODE)
        self.set_to_config(do_save_none, root_section, config, 'SUPERSEED',  self.SUPERSEED)

        self.network.save_to_ini(do_save_none, root_section, config)
        self.dataset.save_to_ini(do_save_none, root_section, config)
        self.train.save_to_ini(do_save_none  , root_section, config)
        self.test.save_to_ini(do_save_none   , root_section, config)

        # Writing our configuration file to 'example.cfg'
        with open(out_file, 'wb') as configfile:
            config.write(configfile)

    def load(self, in_file):
        if not os.path.isfile(in_file):
            err_str = 'File not found: {}'.format(in_file)
            self.log.error(err_str)
            raise AssertionError(err_str)

        override_mode = False
        self.log.info('Loading parameters by file: {}'.format(in_file))
        self.set_from_file(in_file, override_mode)

    def override(self, in_file):
        override_mode = True
        self.log.info('Overriding parameters by file: {}'.format(in_file))
        self.set_from_file(in_file, override_mode)

    def set_from_file(self, in_file, override_mode):
        if not os.path.isfile(in_file):
            err_str = 'File not found: {}'.format(in_file)
            self.log.error(err_str)
            raise AssertionError(err_str)

        parser = configparser.ConfigParser()
        parser._interpolation = configparser.ExtendedInterpolation()
        parser.optionxform = str

        parser.read(in_file)

        root_section = 'self'
        self.parse_from_config(self, override_mode, root_section, parser, 'DEBUG_MODE', bool)
        self.parse_from_config(self, override_mode, root_section, parser, 'SUPERSEED', int)

        self.network.set_from_file(override_mode, root_section, parser)
        self.dataset.set_from_file(override_mode, root_section, parser)
        self.train.set_from_file(override_mode,   root_section, parser)
        self.test.set_from_file(override_mode,    root_section, parser)

class ParametersNetwork(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersNetwork, self).__init__()

        self.ARCHITECTURE = None                       # string: name of architecture. e.g. Wide-Resnet-28-10
        self.DEVICE = None                             # String: device type (tensorflow), e.g. "/gpu:0"
        self.NUM_CLASSES = None                        # integer: number of classes at the output of the classifier
        self.IMAGE_HEIGHT = None                       # integer: Image height at network input
        self.IMAGE_WIDTH = None                        # integer: Image width at network input
        self.NUM_RESIDUAL_UNITS = None                 # integer: number of residual modules in a ResNet model
        self.EMBEDDING_DIMS = None                     # integer: number of fully connected neurons before classifier
        self.BATCH_NORMALIZE_EMBEDDING = None          # boolean: whether or not to apply batch normalization before the embedding activation
        self.NORMALIZE_EMBEDDING = None                # boolean: whether or not to normalize the embedded space
        self.RESNET_FILTERS = None                     # numpy: the number of filters in the resnet bloks

        self.pre_processing = ParametersNetworkPreProcessing()
        self.system         = ParametersNetworkSystem()
        self.optimization   = ParametersNetworkOptimization()

        self._freeze()

    def name(self):
        return 'network'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'ARCHITECTURE'              , self.ARCHITECTURE)
        self.set_to_config(do_save_none, section_name, config, 'DEVICE'                    , self.DEVICE)
        self.set_to_config(do_save_none, section_name, config, 'NUM_CLASSES'               , self.NUM_CLASSES)
        self.set_to_config(do_save_none, section_name, config, 'IMAGE_HEIGHT'              , self.IMAGE_HEIGHT)
        self.set_to_config(do_save_none, section_name, config, 'IMAGE_WIDTH'               , self.IMAGE_WIDTH)
        self.set_to_config(do_save_none, section_name, config, 'NUM_RESIDUAL_UNITS'        , self.NUM_RESIDUAL_UNITS)
        self.set_to_config(do_save_none, section_name, config, 'EMBEDDING_DIMS'            , self.EMBEDDING_DIMS)
        self.set_to_config(do_save_none, section_name, config, 'BATCH_NORMALIZE_EMBEDDING' , self.BATCH_NORMALIZE_EMBEDDING)
        self.set_to_config(do_save_none, section_name, config, 'NORMALIZE_EMBEDDING'       , self.NORMALIZE_EMBEDDING)
        self.set_to_config(do_save_none, section_name, config, 'RESNET_FILTERS'            , self.RESNET_FILTERS)

        self.pre_processing.save_to_ini(do_save_none, section_name, config)
        self.system.save_to_ini(do_save_none, section_name, config)
        self.optimization.save_to_ini(do_save_none, section_name, config)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())

        self.parse_from_config(self, override_mode, section_name, parser, 'ARCHITECTURE'              , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'DEVICE'                    , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'NUM_CLASSES'               , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'IMAGE_HEIGHT'              , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'IMAGE_WIDTH'               , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'NUM_RESIDUAL_UNITS'        , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'EMBEDDING_DIMS'            , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'BATCH_NORMALIZE_EMBEDDING' , bool)
        self.parse_from_config(self, override_mode, section_name, parser, 'NORMALIZE_EMBEDDING'       , bool)
        self.parse_from_config(self, override_mode, section_name, parser, 'RESNET_FILTERS'            , np.array)

        self.pre_processing.set_from_file(override_mode, section_name, parser)
        self.system.set_from_file(override_mode, section_name, parser)
        self.optimization.set_from_file(override_mode, section_name, parser)

class ParametersDataset(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersDataset, self).__init__()

        self.DATASET_NAME = None                             # string, dataset name. e.g.: "cifar10"
        self.TRAIN_SET_SIZE = None                           # integer: train set size
        self.VALIDATION_SET_SIZE = None                      # integer: validation set size
        self.TEST_SET_SIZE = None                            # integer: test set size
        self.TRAIN_VALIDATION_MAP_REF = None                 # string: path to a reference train-validation mapping
        self.CLUSTERS = None                                 # integer: number of new clusters when updating active pool
        self.INIT_SIZE = None                                # integer: the initial pool size when dataset constructs
        self.CAP = None                                      # integer: maximum number of labels in active training

        self._freeze()

    def name(self):
        return 'dataset'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'DATASET_NAME'            , self.DATASET_NAME)
        self.set_to_config(do_save_none, section_name, config, 'TRAIN_SET_SIZE'          , self.TRAIN_SET_SIZE)
        self.set_to_config(do_save_none, section_name, config, 'VALIDATION_SET_SIZE'     , self.VALIDATION_SET_SIZE)
        self.set_to_config(do_save_none, section_name, config, 'TEST_SET_SIZE'           , self.TEST_SET_SIZE)
        self.set_to_config(do_save_none, section_name, config, 'TRAIN_VALIDATION_MAP_REF', self.TRAIN_VALIDATION_MAP_REF)
        self.set_to_config(do_save_none, section_name, config, 'CLUSTERS'                , self.CLUSTERS)
        self.set_to_config(do_save_none, section_name, config, 'INIT_SIZE'               , self.INIT_SIZE)
        self.set_to_config(do_save_none, section_name, config, 'CAP'                     , self.CAP)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'DATASET_NAME'            , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'TRAIN_SET_SIZE'          , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'VALIDATION_SET_SIZE'     , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'TEST_SET_SIZE'           , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'TRAIN_VALIDATION_MAP_REF', str)
        self.parse_from_config(self, override_mode, section_name, parser, 'CLUSTERS'                , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'INIT_SIZE'               , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'CAP'                     , int)

class ParametersTrain(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersTrain, self).__init__()

        self.data_augmentation = ParametersTrainDataAugmentation()
        self.train_control     = ParametersTrainControl()

        self._freeze()

    def name(self):
        return 'train'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.data_augmentation.save_to_ini(do_save_none, section_name, config)
        self.train_control.save_to_ini(do_save_none    , section_name, config)

    def set_from_file(self,override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.data_augmentation.set_from_file(override_mode, section_name, parser)
        self.train_control.set_from_file(override_mode    , section_name, parser)

class ParametersNetworkSystem(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersNetworkSystem, self).__init__()

        self.RELU_LEAKINESS    = None                 # float: The leakiness of the ReLU activation
        self.DROPOUT_KEEP_PROB = None                 # float: The probability to keep neurons after the global pooling

        self._freeze()

    def name(self):
        return 'system'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'RELU_LEAKINESS'   , self.RELU_LEAKINESS)
        self.set_to_config(do_save_none, section_name, config, 'DROPOUT_KEEP_PROB', self.DROPOUT_KEEP_PROB)


    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'RELU_LEAKINESS'   , float)
        self.parse_from_config(self, override_mode, section_name, parser, 'DROPOUT_KEEP_PROB', float)

class ParametersNetworkOptimization(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersNetworkOptimization, self).__init__()

        self.LEARNING_RATE         = None    # float: e.g. 0.1
        self.XENTROPY_RATE         = None    # float: e.g. 1.0
        self.DML_MARGIN_MULTIPLIER = None    # float: the DML margin to calculate the loss. e.g. 1.0
        self.WEIGHT_DECAY_RATE     = None    # float: e.g. 0.00078125
        self.OPTIMIZER             = None    # string: name of optimizer, e.g. 'MOM'

        self._freeze()

    def name(self):
        return 'optimization'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'LEARNING_RATE'        , self.LEARNING_RATE)
        self.set_to_config(do_save_none, section_name, config, 'XENTROPY_RATE'        , self.XENTROPY_RATE)
        self.set_to_config(do_save_none, section_name, config, 'DML_MARGIN_MULTIPLIER', self.DML_MARGIN_MULTIPLIER)
        self.set_to_config(do_save_none, section_name, config, 'WEIGHT_DECAY_RATE'    , self.WEIGHT_DECAY_RATE)
        self.set_to_config(do_save_none, section_name, config, 'OPTIMIZER'            , self.OPTIMIZER)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'LEARNING_RATE'        , float)
        self.parse_from_config(self, override_mode, section_name, parser, 'XENTROPY_RATE'        , float)
        self.parse_from_config(self, override_mode, section_name, parser, 'DML_MARGIN_MULTIPLIER', float)
        self.parse_from_config(self, override_mode, section_name, parser, 'WEIGHT_DECAY_RATE'    , float)
        self.parse_from_config(self, override_mode, section_name, parser, 'OPTIMIZER'            , str)

class ParametersTrainDataAugmentation(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersTrainDataAugmentation, self).__init__()

        #FIXME(gilad): incorporate into dataset wrapper
        self.FLIP_IMAGE         = None   # boolean: whether to randomly flip images due to augmentation
        self.DRIFT_X            = None   # int: drift x for augmentation, e.g. 45
        self.DRIFT_Y            = None   # int: drift y for image augmentation, e.g. 20

        self._freeze()

    def name(self):
        return 'data_augmentation'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'FLIP_IMAGE'        , self.FLIP_IMAGE)
        self.set_to_config(do_save_none, section_name, config, 'DRIFT_X'           , self.DRIFT_X)
        self.set_to_config(do_save_none, section_name, config, 'DRIFT_Y'           , self.DRIFT_Y)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self,override_mode, section_name, parser,'FLIP_IMAGE'        , bool)
        self.parse_from_config(self,override_mode, section_name, parser,'DRIFT_X'           , int)
        self.parse_from_config(self,override_mode, section_name, parser,'DRIFT_Y'           , int)

class ParametersTrainControl(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersTrainControl, self).__init__()

        self.TRAINER               = None  # string: trainer to use. e.g. passive
        self.TRAIN_BATCH_SIZE      = None  # integer: batch size for training, e.g. 200
        self.EVAL_BATCH_SIZE       = None  # integer: batch size for evaluating, e.g. 2200
        self.ROOT_DIR              = None  # string: path to root dir that contain train/validation dirs
        self.TRAIN_DIR             = None  # string: path to train dir
        self.EVAL_DIR              = None  # string: path to validation dir
        self.TEST_DIR              = None  # string: path to test dir
        self.PREDICTION_DIR        = None  # string: path to prediction dir
        self.CHECKPOINT_DIR        = None  # string: path to checkpoint dir
        self.CHECKPOINT_REF        = None  # string: path to a checkpoint reference file to load
        self.SUMMARY_STEPS         = None  # integer: training steps to collect summary
        self.CHECKPOINT_SECS       = None  # integer: number of seconds to save new checkpoint
        self.CHECKPOINT_STEPS      = None  # np.array: global_steps where the parameters are saved
        self.LAST_STEP             = None  # integer: number of training steps before the training session stops.
        self.LOGGER_STEPS          = None  # integer: number of training steps to output log string to shell
        self.EVAL_STEPS            = None  # integer: number of training steps from one evaluation to the next
        self.TEST_STEPS            = None  # integer: number of training steps from one test to the next
        self.RETENTION_SIZE        = None  # integer: the number of last scores to remember
        self.MIN_LEARNING_RATE     = None  # float: minimal learning rate before choosing new labels in active training
        self.SKIP_FIRST_EVALUATION = None  # boolean: whether or not to skip the first evaluation in the training
        self.PCA_REDUCTION         = None  # boolean: whether or not to use PCA reduction
        self.PCA_EMBEDDING_DIMS    = None  # integer: PCA dimensions
        self.ANNOTATION_RULE       = None  # The rule for adding new annotations in active learning
        self.STEPS_FOR_NEW_ANNOTATIONS = None # integer: global steps to add annotations
        self.INIT_AFTER_ANNOT      = None  # Whether or not to initialize network weights after annotation phase
        self.ACTIVE_SELECTION_CRITERION = None  # string: the method for the active learning

        self.learning_rate_setter     = ParametersTrainControlLearningRateSetter()
        self.margin_multiplier_setter = ParametersTrainControlMarginMultiplierSetter()

        self._freeze()

    def name(self):
        return 'train_control'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'TRAINER'              , self.TRAINER)
        self.set_to_config(do_save_none, section_name, config, 'TRAIN_BATCH_SIZE'     , self.TRAIN_BATCH_SIZE)
        self.set_to_config(do_save_none, section_name, config, 'EVAL_BATCH_SIZE'      , self.EVAL_BATCH_SIZE)
        self.set_to_config(do_save_none, section_name, config, 'ROOT_DIR'             , self.ROOT_DIR)
        self.set_to_config(do_save_none, section_name, config, 'TRAIN_DIR'            , self.TRAIN_DIR)
        self.set_to_config(do_save_none, section_name, config, 'EVAL_DIR'             , self.EVAL_DIR)
        self.set_to_config(do_save_none, section_name, config, 'TEST_DIR'             , self.TEST_DIR)
        self.set_to_config(do_save_none, section_name, config, 'PREDICTION_DIR'       , self.PREDICTION_DIR)
        self.set_to_config(do_save_none, section_name, config, 'CHECKPOINT_DIR'       , self.CHECKPOINT_DIR)
        self.set_to_config(do_save_none, section_name, config, 'CHECKPOINT_REF'       , self.CHECKPOINT_REF)
        self.set_to_config(do_save_none, section_name, config, 'SUMMARY_STEPS'        , self.SUMMARY_STEPS)
        self.set_to_config(do_save_none, section_name, config, 'CHECKPOINT_SECS'      , self.CHECKPOINT_SECS)
        self.set_to_config(do_save_none, section_name, config, 'CHECKPOINT_STEPS'     , self.CHECKPOINT_STEPS)
        self.set_to_config(do_save_none, section_name, config, 'LAST_STEP'            , self.LAST_STEP)
        self.set_to_config(do_save_none, section_name, config, 'LOGGER_STEPS'         , self.LOGGER_STEPS)
        self.set_to_config(do_save_none, section_name, config, 'EVAL_STEPS'           , self.EVAL_STEPS)
        self.set_to_config(do_save_none, section_name, config, 'TEST_STEPS'           , self.TEST_STEPS)
        self.set_to_config(do_save_none, section_name, config, 'RETENTION_SIZE'       , self.RETENTION_SIZE)
        self.set_to_config(do_save_none, section_name, config, 'MIN_LEARNING_RATE'    , self.MIN_LEARNING_RATE)
        self.set_to_config(do_save_none, section_name, config, 'SKIP_FIRST_EVALUATION', self.SKIP_FIRST_EVALUATION)
        self.set_to_config(do_save_none, section_name, config, 'PCA_REDUCTION'        , self.PCA_REDUCTION)
        self.set_to_config(do_save_none, section_name, config, 'PCA_EMBEDDING_DIMS'   , self.PCA_EMBEDDING_DIMS)
        self.set_to_config(do_save_none, section_name, config, 'ANNOTATION_RULE'      , self.ANNOTATION_RULE)
        self.set_to_config(do_save_none, section_name, config, 'STEPS_FOR_NEW_ANNOTATIONS' , self.STEPS_FOR_NEW_ANNOTATIONS)
        self.set_to_config(do_save_none, section_name, config, 'INIT_AFTER_ANNOT'     , self.INIT_AFTER_ANNOT)
        self.set_to_config(do_save_none, section_name, config, 'ACTIVE_SELECTION_CRITERION', self.ACTIVE_SELECTION_CRITERION)

        self.learning_rate_setter.save_to_ini(do_save_none, section_name, config)
        self.margin_multiplier_setter.save_to_ini(do_save_none, section_name, config)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'TRAINER'              , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'TRAIN_BATCH_SIZE'     , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'EVAL_BATCH_SIZE'      , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'ROOT_DIR'             , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'TRAIN_DIR'            , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'EVAL_DIR'             , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'TEST_DIR'             , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'PREDICTION_DIR'       , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'CHECKPOINT_DIR'       , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'CHECKPOINT_REF'       , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'SUMMARY_STEPS'        , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'CHECKPOINT_SECS'      , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'CHECKPOINT_STEPS'     , np.array)
        self.parse_from_config(self, override_mode, section_name, parser, 'LAST_STEP'            , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'LOGGER_STEPS'         , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'EVAL_STEPS'           , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'TEST_STEPS'           , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'RETENTION_SIZE'       , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'MIN_LEARNING_RATE'    , float)
        self.parse_from_config(self, override_mode, section_name, parser, 'SKIP_FIRST_EVALUATION', bool)
        self.parse_from_config(self, override_mode, section_name, parser, 'PCA_REDUCTION'        , bool)
        self.parse_from_config(self, override_mode, section_name, parser, 'PCA_EMBEDDING_DIMS'   , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'ANNOTATION_RULE'      , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'STEPS_FOR_NEW_ANNOTATIONS' , np.array)
        self.parse_from_config(self, override_mode, section_name, parser, 'INIT_AFTER_ANNOT'     , bool)
        self.parse_from_config(self, override_mode, section_name, parser, 'ACTIVE_SELECTION_CRITERION', str)

        self.learning_rate_setter.set_from_file(override_mode, section_name, parser)
        self.margin_multiplier_setter.set_from_file(override_mode, section_name, parser)

class ParametersNetworkPreProcessing(parser_utils.FrozenClass):
    def __init__(self):
        parser_utils.FrozenClass.__init__(self)

        # Not in use. Might be relevant someday if I wish to run with multiple preprocessing pipes.
        self.PREPROCESSOR = None     # string: preprocessor name. e.g. preprocessor_drift_flip

        self._freeze()

    def name(self):
        return 'pre_processing'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'PREPROCESSOR', self.PREPROCESSOR)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'PREPROCESSOR', str)

class ParametersTrainControlLearningRateSetter(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersTrainControlLearningRateSetter, self).__init__()

        self.LEARNING_RATE_SETTER          = None  # string: Name of the learning rate setter
        self.SCHEDULED_STEPS               = None  # np.array: the epochs in which the learning rate is decreased
        self.USE_FIXED_EPOCHS              = None  # boolean: use epochs instead of global steps
        self.SCHEDULED_LEARNING_RATES      = None  # np.array: the updated learning rates at each SCHEDULED_STEPS
        self.LR_DECAY_REFRACTORY_STEPS     = None  # integer: number of training steps after decaying the learning
                                                   # rate in which no new decay can be utilized
        self.LEARNING_RATE_RESET           = None  # float: reset value of learning rate

        self._freeze()

    def name(self):
        return 'learning_rate_setter'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'LEARNING_RATE_SETTER'     , self.LEARNING_RATE_SETTER)
        self.set_to_config(do_save_none, section_name, config, 'SCHEDULED_STEPS'          , self.SCHEDULED_STEPS)
        self.set_to_config(do_save_none, section_name, config, 'USE_FIXED_EPOCHS'         , self.USE_FIXED_EPOCHS)
        self.set_to_config(do_save_none, section_name, config, 'SCHEDULED_LEARNING_RATES' , self.SCHEDULED_LEARNING_RATES)
        self.set_to_config(do_save_none, section_name, config, 'LR_DECAY_REFRACTORY_STEPS', self.LR_DECAY_REFRACTORY_STEPS)
        self.set_to_config(do_save_none, section_name, config, 'LEARNING_RATE_RESET'      , self.LEARNING_RATE_RESET)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'LEARNING_RATE_SETTER'        , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'SCHEDULED_STEPS'             , np.array)
        self.parse_from_config(self, override_mode, section_name, parser, 'USE_FIXED_EPOCHS'            , bool)
        self.parse_from_config(self, override_mode, section_name, parser, 'SCHEDULED_LEARNING_RATES'    , np.array)
        self.parse_from_config(self, override_mode, section_name, parser, 'LR_DECAY_REFRACTORY_STEPS'   , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'LEARNING_RATE_RESET'         , float)

class ParametersTrainControlMarginMultiplierSetter(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersTrainControlMarginMultiplierSetter, self).__init__()

        self.MM_DECAY_REFRACTORY_STEPS     = None  # integer: number of training steps after decaying the margin
                                                   # multiplier in which no new decay can be utilized
        self._freeze()

    def name(self):
        return 'margin_multiplier_setter'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'MM_DECAY_REFRACTORY_STEPS', self.MM_DECAY_REFRACTORY_STEPS)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'MM_DECAY_REFRACTORY_STEPS'   , int)

class ParametersTest(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersTest, self).__init__()

        self.test_control = ParametersTestControl()
        self.ensemble     = ParametersTestEnsemble()

        self._freeze()

    def name(self):
        return 'test'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.test_control.save_to_ini(do_save_none    , section_name, config)
        self.ensemble.save_to_ini(do_save_none        , section_name, config)

    def set_from_file(self,override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.test_control.set_from_file(override_mode    , section_name, parser)
        self.ensemble.set_from_file(override_mode        , section_name, parser)

class ParametersTestControl(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersTestControl, self).__init__()

        self.TESTER                = None  # string: tester to use. e.g. knn_classifier
        self.DECISION_METHOD       = None  # string: The decision method for the classification
        self.CHECKPOINT_FILE       = None  # string: The checkpoint file name to read. e.g. model.ckpt-50000
        self.KNN_NEIGHBORS         = None  # integer: number of knn neighbors, e.g. 200
        self.KNN_NORM              = None  # integer: knn norm. L1 or L2, e.g. 2
        self.KNN_WEIGHTS           = None  # string: either 'distance' or 'uniform'
        self.KNN_JOBS              = None  # integer: number of KNN n_jobs, should be the number of available CPUs
        self.DUMP_NET              = None  # boolean: whether or not to dump the net signals to disk
        self.LOAD_FROM_DISK        = None  # boolean: whether or not to load the network data from the .npy files

        self._freeze()

    def name(self):
        return 'test_control'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'TESTER'               , self.TESTER)
        self.set_to_config(do_save_none, section_name, config, 'DECISION_METHOD'      , self.DECISION_METHOD)
        self.set_to_config(do_save_none, section_name, config, 'CHECKPOINT_FILE'      , self.CHECKPOINT_FILE)
        self.set_to_config(do_save_none, section_name, config, 'KNN_NEIGHBORS'        , self.KNN_NEIGHBORS)
        self.set_to_config(do_save_none, section_name, config, 'KNN_NORM'             , self.KNN_NORM)
        self.set_to_config(do_save_none, section_name, config, 'KNN_WEIGHTS'          , self.KNN_WEIGHTS)
        self.set_to_config(do_save_none, section_name, config, 'KNN_JOBS'             , self.KNN_JOBS)
        self.set_to_config(do_save_none, section_name, config, 'DUMP_NET'             , self.DUMP_NET)
        self.set_to_config(do_save_none, section_name, config, 'LOAD_FROM_DISK'       , self.LOAD_FROM_DISK)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'TESTER'          , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'DECISION_METHOD' , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'CHECKPOINT_FILE' , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'KNN_NEIGHBORS'   , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'KNN_NORM'        , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'KNN_WEIGHTS'     , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'KNN_JOBS'        , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'DUMP_NET'        , bool)
        self.parse_from_config(self, override_mode, section_name, parser, 'LOAD_FROM_DISK'  , bool)

class ParametersTestEnsemble(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersTestEnsemble, self).__init__()

        self.LOG_DIR_LIST          = None  # list: root dirs that make the ensemble

        self._freeze()

    def name(self):
        return 'ensemble'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'LOG_DIR_LIST'               , self.LOG_DIR_LIST)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'LOG_DIR_LIST'          , list)
