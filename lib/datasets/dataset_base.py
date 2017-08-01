from abc import ABCMeta, abstractmethod
import lib.logger.logger as logger

class DataSetBase(object):
    """Base class that holds a specific dataset: train or validation - not both"""
    __metaclass__ = ABCMeta

    def __init__(self, name, prm, preprocessor):
        self.name = name
        self.prm = prm
        self.preprocessor = preprocessor
        self.log = logger.get_logger(name)

        if 'train' in name:
            self.size          = self.prm.dataset.TRAIN_SET_SIZE
            self.batch_size    = self.prm.train.train_control.TRAIN_BATCH_SIZE
            self.images_dir    = self.prm.dataset.TRAIN_IMAGES_DIR
            self.labels_file   = self.prm.dataset.TRAIN_LABELS_FILE
            self.to_preprocess = True
        elif 'validation' in name:
            self.size          = self.prm.dataset.VALIDATION_SET_SIZE
            self.batch_size    = self.prm.train.train_control.EVAL_BATCH_SIZE
            self.images_dir    = self.prm.dataset.VALIDATION_IMAGES_DIR
            self.labels_file   = self.prm.dataset.VALIDATION_LABELS_FILE
            self.to_preprocess = False
        else:
            err_str = self.__str__() + ': __init__: name ({}) must include train or validation'.format(name)
            self.log.error(err_str)
            raise NameError(err_str)

        self.pool = None  # list of indices which can be chosen for a batch

    def __str__(self):
        return self.name

    def print_stats(self):
        pass

    @abstractmethod
    def get_mini_batch(self, batch_size=None):
        pass

    def pool_size(self):
        return len(self.pool)

    def initialize_pool(self):
        """Must be called immidiately after __init__"""
        self.pool = range(self.size)  # list of indices which can be chosen for a batch
