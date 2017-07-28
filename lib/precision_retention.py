import lib.logger.logger as logger

class PrecisionRetention(object):

    def __init__(self, name, prm):
        self.name = name
        self.prm = prm
        self.log = logger.get_logger(name)

        self.precision_retention_size = self.prm.train.train_control.PRECISION_RETENTION_SIZE

        self.best_precision = 0.0
        self.precision_memory = [0.0] * self.precision_retention_size

    def __str__(self):
        return self.name

    def print_stats(self):
        self.log.info(self.__str__() + 'parameters:')
        self.log.info(' PRECISION_RETENTION_SIZE: {}'.format(self.precision_retention_size))

    def set_best_precision(self, precision):
        self.best_precision = precision
        self.log.info('Setting new best precision: {}'.format(precision))

    def get_best_precision(self):
        return self.best_precision

    def print_memory(self):
        self.log.info('Current precision memory is:\n\t{}'.format(self.precision_memory))

    def add_precision(self, precision):
        self.precision_memory = self.precision_memory[1:] + [precision]
        if precision > self.best_precision:
            self.set_best_precision(precision)

    def is_precision_stuck(self):
        return all(val < self.get_best_precision() for val in self.precision_memory)



