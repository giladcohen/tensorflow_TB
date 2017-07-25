from abc import ABCMeta, abstractmethod
import lib.logger.logger as logger

class PreProcessorBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, prm):
        self.name = name
        self.prm = prm
        self.log = logger.get_logger(name)
        self.preprocessor       = self.prm.network.pre_processing.PREPROCESSOR
        self.data_augmentation  = self.prm.train.data_augmentation.DATA_AUGMENTATION
        self.label_augmentation = self.prm.train.data_augmentation.LABEL_AUGMENTATION
        self.flip_image         = self.prm.train.data_augmentation.FLIP_IMAGE
        self.drift_x            = self.prm.train.data_augmentation.DRIFT_X
        self.drift_y            = self.prm.train.data_augmentation.DRIFT_Y

        self.assert_config()

    def print_stats(self):
        """print basic preprocessor parameters"""
        self.log.info('PreProcessor parameters:')
        self.log.info(' PREPROCESSOR: {}'.format(self.preprocessor))
        self.log.info(' DATA_AUGMENTATION: {}'.format(self.data_augmentation))
        self.log.info(' LABEL_AUGMENTATION: {}'.format(self.label_augmentation))
        self.log.info(' FLIP_IMAGE: {}'.format(self.flip_image))
        self.log.info(' DRIFT_X: {}'.format(self.drift_x))
        self.log.info(' DRIFT_Y: {}'.format(self.drift_y))

    def __str__(self):
        return self.name

    def assert_config(self):
        if self.label_augmentation and not self.data_augmentation:
            err_str = 'assert_config: LABEL_AUGMENTATION cannot be set to True with DATA_AUGMENTATION=False'
            self.log.error(err_str)
            raise ValueError(err_str)
        if self.drift_x < 0 or self.drift_y < 0:
            err_str = 'assert_config: DRIFT_X ({}) and DRIFT_Y ({}) should both be positive'.format(self.drift_x, self.drift_y)
            self.log.error(err_str)
            raise ValueError(err_str)

    @abstractmethod
    def process(self, images, labels):
        """processing batch of images and labels"""
        if images.shape[0] == labels.shape[0]:
            err_str = 'process: number of images ({}) does not match number of labels ({})'.format(images.shape[0], labels.shape[0])
            self.log.error(err_str)
            raise AssertionError(err_str)


