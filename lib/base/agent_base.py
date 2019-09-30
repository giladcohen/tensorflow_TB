import tensorflow_TB.lib.logger.logger as logger
from abc import ABCMeta


class AgentBase(object):
    """Base class for all the agents in the train bench:
    Model
    Trainer
    Dataset
    Hooks
    et cetera"""
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name
        self.log = logger.get_logger(name)

    def __str__(self):
        return self.name

    def print_stats(self):
        pass
