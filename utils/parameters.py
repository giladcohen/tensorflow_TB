from __future__ import absolute_import

import os
import configparser
import utils.ini_parsers as parser_utils


class Parameters(parser_utils.FrozenClass):
    def __init__(self):
        super(Parameters, self).__init__()

        self.DEBUG_MODE = None                   # boolean
        self.network = ParametersNetwork()
        self.dataset = ParametersDataset()
        self.train   = ParametersTrain()

        self._freeze()  # no new attributes after this point.

    def save(self, out_file, do_save_none=True):
        """
        Save  parameters. Undefined parameters (with None value) are saved with value = None
        :param out_file: Path to file to be saved
        """
        #----------------------------------
        # dataset section
        #----------------------------------
        self.log.info('Configuration saved to file: {}'.format(out_file))

        # add the settings to the structure of the file, and lets write it out...
        config = configparser.ConfigParser()
        config._interpolation = configparser.ExtendedInterpolation()

        config.optionxform = str

        root_section = 'self'
        config.add_section(root_section)
        self.set_to_config(do_save_none, root_section, config, 'DEBUG_MODE', self.DEBUG_MODE)

        self.network.save_to_ini(do_save_none, root_section, config)
        self.dataset.save_to_ini(do_save_none, root_section, config)
        self.train.save_to_ini(do_save_none,   root_section, config)

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

        self.network.set_from_file(override_mode, root_section, parser)
        self.dataset.set_from_file(override_mode, root_section, parser)
        self.train.set_from_file(override_mode,   root_section, parser)

class ParametersNetwork(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersNetwork, self).__init__()

        self.ARCHITECTURE = None                       # string: name of architecture. e.g. Wide-Resnet-28-10
        self.DEVICE = None                             # String: device type (tensorflow), e.g. "/gpu:0"
        self.NUM_CLASSES = None                        # integer: number of classes at the output of the classifier
        self.IMAGE_HEIGHT = None                       # integer: Image height at network input
        self.IMAGE_WIDTH = None                        # integer: Image width at network input
        self.NUM_RESIDUAL_UNITS = None                 # integer: number of residual modules in a ResNet model

        self.pre_processing = ParametersNetworkPreProcessing()
        self.system         = ParametersNetworkSystem()
        self.optimization   = ParametersNetworkOptimization()

        self._freeze()

    def name(self):
        return 'network'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'ARCHITECTURE'       , self.ARCHITECTURE)
        self.set_to_config(do_save_none, section_name, config, 'DEVICE'             , self.DEVICE)
        self.set_to_config(do_save_none, section_name, config, 'NUM_CLASSES'        , self.NUM_CLASSES)
        self.set_to_config(do_save_none, section_name, config, 'IMAGE_HEIGHT'       , self.IMAGE_HEIGHT)
        self.set_to_config(do_save_none, section_name, config, 'IMAGE_WIDTH'        , self.IMAGE_WIDTH)
        self.set_to_config(do_save_none, section_name, config, 'NUM_RESIDUAL_UNITS' , self.NUM_RESIDUAL_UNITS)

        self.pre_processing.save_to_ini(do_save_none, section_name, config)
        self.system.save_to_ini(do_save_none, section_name, config)
        self.optimization.save_to_ini(do_save_none, section_name, config)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())

        self.parse_from_config(self, override_mode, section_name, parser, 'ARCHITECTURE'       , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'DEVICE'             , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'NUM_CLASSES'        , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'IMAGE_HEIGHT'       , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'IMAGE_WIDTH'        , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'NUM_RESIDUAL_UNITS' , int)

        self.pre_processing.set_from_file(override_mode, section_name, parser)
        self.system.set_from_file(override_mode, section_name, parser)
        self.optimization.set_from_file(override_mode, section_name, parser)

class ParametersDataset(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersDataset, self).__init__()

        self.DATASET_NAME = None                             # string, dataset name. e.g.: "cifar10"
        self.DATASET_DIR = None                              # string, dataset directory
        self.TRAIN_SET_SIZE = None                           # integer: train set size
        self.VALIDATION_SET_SIZE = None                      # integer: validation set size
        self.TRAIN_IMAGES_DIR = None                         # string: path to train images dir
        self.TRAIN_LABELS_FILE = None                        # string: path to train labels file
        self.VALIDATION_IMAGES_DIR = None                    # string: path to validation images dir
        self.VALIDATION_LABELS_FILE = None                   # string: path to validation labels file
        self.N_CLUSTERS = None                               # integer: number of new clusters when updating active pool
        self.CAP = None                                      # integer: maximum number of labels in active training

        self._freeze()

    def name(self):
        return 'dataset'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'DATASET_NAME'          , self.DATASET_NAME)
        self.set_to_config(do_save_none, section_name, config, 'DATASET_DIR'           , self.DATASET_DIR)
        self.set_to_config(do_save_none, section_name, config, 'TRAIN_SET_SIZE'        , self.TRAIN_SET_SIZE)
        self.set_to_config(do_save_none, section_name, config, 'VALIDATION_SET_SIZE'   , self.VALIDATION_SET_SIZE)
        self.set_to_config(do_save_none, section_name, config, 'TRAIN_IMAGES_DIR'      , self.TRAIN_IMAGES_DIR)
        self.set_to_config(do_save_none, section_name, config, 'TRAIN_LABELS_FILE'     , self.TRAIN_LABELS_FILE)
        self.set_to_config(do_save_none, section_name, config, 'VALIDATION_IMAGES_DIR' , self.VALIDATION_IMAGES_DIR)
        self.set_to_config(do_save_none, section_name, config, 'VALIDATION_LABELS_FILE', self.VALIDATION_LABELS_FILE)
        self.set_to_config(do_save_none, section_name, config, 'N_CLUSTERS'            , self.N_CLUSTERS)
        self.set_to_config(do_save_none, section_name, config, 'CAP'                   , self.CAP)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'DATASET_NAME'          , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'DATASET_DIR'           , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'TRAIN_SET_SIZE'        , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'VALIDATION_SET_SIZE'   , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'TRAIN_IMAGES_DIR'      , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'TRAIN_LABELS_FILE'     , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'VALIDATION_IMAGES_DIR' , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'VALIDATION_LABELS_FILE', str)
        self.parse_from_config(self, override_mode, section_name, parser, 'N_CLUSTERS'            , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'CAP'                   , int)

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

        self.RELU_LEAKINESS = None                 # float: The leakiness of the ReLU activation

        self._freeze()

    def name(self):
        return 'system'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'RELU_LEAKINESS', self.RELU_LEAKINESS)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'RELU_LEAKINESS', float)

class ParametersNetworkOptimization(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersNetworkOptimization, self).__init__()

        self.LEARNING_RATE     =  None   # float: e.g. 0.1
        self.XENTROPY_RATE     = None    # float: e.g. 1.0
        self.WEIGHT_DECAY_RATE = None    # float: e.g. 0.00078125
        self.OPTIMIZER         = None    # string: name of optimizer, e.g. 'MOM'

        self._freeze()

    def name(self):
        return 'optimization'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'LEARNING_RATE'    , self.LEARNING_RATE)
        self.set_to_config(do_save_none, section_name, config, 'XENTROPY_RATE'    , self.XENTROPY_RATE)
        self.set_to_config(do_save_none, section_name, config, 'WEIGHT_DECAY_RATE', self.WEIGHT_DECAY_RATE)
        self.set_to_config(do_save_none, section_name, config, 'OPTIMIZER'        , self.OPTIMIZER)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'LEARNING_RATE'    , float)
        self.parse_from_config(self, override_mode, section_name, parser, 'XENTROPY_RATE'    , float)
        self.parse_from_config(self, override_mode, section_name, parser, 'WEIGHT_DECAY_RATE', float)
        self.parse_from_config(self, override_mode, section_name, parser, 'OPTIMIZER'        , str)

class ParametersTrainDataAugmentation(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersTrainDataAugmentation, self).__init__()

        self.DATA_AUGMENTATION  = None   # boolean: whether to do data augmentation in pre-processing of images
        self.LABEL_AUGMENTATION = None   # boolean: whether to do label augmentation (on top of data augmentation) in pre-processing
        self.FLIP_IMAGE         = None   # boolean: whether to randomly flip images due to augmentation
        self.DRIFT_X            = None   # int: drift x for augmentation, e.g. 45
        self.DRIFT_Y            = None   # int: drift y for image augmentation, e.g. 20

        self._freeze()

    def name(self):
        return 'data_augmentation'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'DATA_AUGMENTATION' , self.DATA_AUGMENTATION)
        self.set_to_config(do_save_none, section_name, config, 'LABEL_AUGMENTATION', self.LABEL_AUGMENTATION)
        self.set_to_config(do_save_none, section_name, config, 'FLIP_IMAGE'        , self.FLIP_IMAGE)
        self.set_to_config(do_save_none, section_name, config, 'DRIFT_X'           , self.DRIFT_X)
        self.set_to_config(do_save_none, section_name, config, 'DRIFT_Y'           , self.DRIFT_Y)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self,override_mode, section_name, parser,'DATA_AUGMENTATION' , bool)
        self.parse_from_config(self,override_mode, section_name, parser,'LABEL_AUGMENTATION', bool)
        self.parse_from_config(self,override_mode, section_name, parser,'FLIP_IMAGE'        , bool)
        self.parse_from_config(self,override_mode, section_name, parser,'DRIFT_X'           , int)
        self.parse_from_config(self,override_mode, section_name, parser,'DRIFT_Y'           , int)

class ParametersTrainControl(parser_utils.FrozenClass):
    def __init__(self):
        super(ParametersTrainControl, self).__init__()

        self.TRAINER          = None  # string: trainer to use. e.g. passive
        self.TRAIN_BATCH_SIZE = None  # integer: batch size for training, e.g. 200
        self.EVAL_BATCH_SIZE  = None  # integer: batch size for evaluating, e.g. 2200
        self.ROOT_DIR         = None  # string: path to root dir that contain train/validation dirs
        self.TRAIN_DIR        = None  # string: path to train dir
        self.EVAL_DIR         = None  # string: path to validation dir
        self.CHECKPOINT_DIR   = None  # string: path to checkpoint dir
        self.SUMMARY_STEPS    = None  # integer: training steps to collect summary
        self.CHECKPOINT_SECS  = None  # integer: number of seconds to save new checkpoint
        self.LOGGER_STEPS     = None  # integer: number of training steps to output log string to shell
        self.EVAL_STEPS       = None  # integer: number of training steps from one evaluation to the next
        self.EVALS_IN_EPOCH   = None  # integer: number of evaluation steps within an epoch
        self.PRECISION_RETENTION_SIZE = None  # integer: the number of last precisions to remember

        self.learning_rate_setter = ParametersTrainControlLearningRateSetter()

        self._freeze()

    def name(self):
        return 'train_control'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'TRAINER'         , self.TRAINER)
        self.set_to_config(do_save_none, section_name, config, 'TRAIN_BATCH_SIZE', self.TRAIN_BATCH_SIZE)
        self.set_to_config(do_save_none, section_name, config, 'EVAL_BATCH_SIZE' , self.EVAL_BATCH_SIZE)
        self.set_to_config(do_save_none, section_name, config, 'ROOT_DIR'        , self.ROOT_DIR)
        self.set_to_config(do_save_none, section_name, config, 'TRAIN_DIR'       , self.TRAIN_DIR)
        self.set_to_config(do_save_none, section_name, config, 'EVAL_DIR'        , self.EVAL_DIR)
        self.set_to_config(do_save_none, section_name, config, 'CHECKPOINT_DIR'  , self.CHECKPOINT_DIR)
        self.set_to_config(do_save_none, section_name, config, 'SUMMARY_STEPS'   , self.SUMMARY_STEPS)
        self.set_to_config(do_save_none, section_name, config, 'CHECKPOINT_SECS' , self.CHECKPOINT_SECS)
        self.set_to_config(do_save_none, section_name, config, 'LOGGER_STEPS'    , self.LOGGER_STEPS)
        self.set_to_config(do_save_none, section_name, config, 'EVAL_STEPS'      , self.EVAL_STEPS)
        self.set_to_config(do_save_none, section_name, config, 'EVALS_IN_EPOCH'  , self.EVALS_IN_EPOCH)
        self.set_to_config(do_save_none, section_name, config, 'PRECISION_RETENTION_SIZE', self.PRECISION_RETENTION_SIZE)

        self.learning_rate_setter.save_to_ini(do_save_none, section_name, config)

    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'TRAINER'         , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'TRAIN_BATCH_SIZE', int)
        self.parse_from_config(self, override_mode, section_name, parser, 'EVAL_BATCH_SIZE' , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'ROOT_DIR'        , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'TRAIN_DIR'       , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'EVAL_DIR'        , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'CHECKPOINT_DIR'  , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'SUMMARY_STEPS'   , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'CHECKPOINT_SECS' , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'LOGGER_STEPS'    , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'EVAL_STEPS'      , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'EVALS_IN_EPOCH'  , int)
        self.parse_from_config(self, override_mode, section_name, parser, 'PRECISION_RETENTION_SIZE', int)

        self.learning_rate_setter.set_from_file(override_mode, section_name, parser)

class ParametersNetworkPreProcessing(parser_utils.FrozenClass):
    def __init__(self):
        parser_utils.FrozenClass.__init__(self)

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
        self.SCHEDULED_EPOCHS              = None  # list: the epochs in which the learning rate is decreased
        self.SCHEDULED_LEARNING_RATES      = None  # list: the updated learning rates at each SCHEDULED_EPOCHS
        self.DECAY_REFRACTORY_STEPS        = None  # integer: number of training steps after decaying the learning
                                                   # rate in which no new decay can be utilized

        self._freeze()

    def name(self):
        return 'learning_rate_setter'

    def save_to_ini(self, do_save_none, txt, config):
        section_name = self.add_section(txt, self.name(), config)
        self.set_to_config(do_save_none, section_name, config, 'LEARNING_RATE_SETTER'     , self.LEARNING_RATE_SETTER)
        self.set_to_config(do_save_none, section_name, config, 'SCHEDULED_EPOCHS'         , self.SCHEDULED_EPOCHS)
        self.set_to_config(do_save_none, section_name, config, 'SCHEDULED_LEARNING_RATES' , self.SCHEDULED_LEARNING_RATES)
        self.set_to_config(do_save_none, section_name, config, 'DECAY_REFRACTORY_STEPS'   , self.DECAY_REFRACTORY_STEPS)


    def set_from_file(self, override_mode, txt, parser):
        section_name = self.add_section(txt, self.name())
        self.parse_from_config(self, override_mode, section_name, parser, 'LEARNING_RATE_SETTER'        , str)
        self.parse_from_config(self, override_mode, section_name, parser, 'SCHEDULED_EPOCHS'            , list)
        self.parse_from_config(self, override_mode, section_name, parser, 'SCHEDULED_LEARNING_RATES'    , list)
        self.parse_from_config(self, override_mode, section_name, parser, 'DECAY_REFRACTORY_STEPS'      , list)
