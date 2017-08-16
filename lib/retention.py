from lib.base.agent_base import AgentBase


class Retention(AgentBase):

    def __init__(self, name, prm):
        super(Retention, self).__init__(name)
        self.prm = prm

        self.retention_size = self.prm.train.train_control.RETENTION_SIZE

        self.best_score      = None  # need to be reset
        self.best_score_step = None  # need to be reset
        self.memory          = None  # need to be reset

        self.reset_memory()

    def print_stats(self):
        self.log.info(self.__str__() + 'parameters:')
        self.log.info(' RETENTION_SIZE: {}'.format(self.retention_size))

    def set_best_score(self, score, global_step):
        """Provide both the new best score value AND the global step when it achieved"""
        self.best_score = score
        self.best_score_step = global_step
        self.log.info('global_step: {}. Setting new best score: {}'.format(global_step, score))

    def get_best_score(self):
        return self.best_score

    def print_memory(self):
        self.log.info('Current score memory is:\n\t{}'.format(self.memory))

    def add_score(self, score, global_step):
        self.memory = self.memory[1:] + [score]
        if score > self.best_score:
            self.set_best_score(score, global_step)

    def is_score_stuck(self):
        return all(val < self.get_best_score() for val in self.memory)

    def reset_memory(self):
        self.log.info('Reseting ' + self.__str__() + ' memory')
        self.best_score = 0.0
        self.best_score_step = 0
        self.memory = [0.0] * self.retention_size
