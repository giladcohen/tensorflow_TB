from lib.trainers.hooks.learning_rate_setter_base import LearningRateSetterBase

class PrecisionDecaySetter(LearningRateSetterBase):
    """Decaying the learning rate based on the precision retention.
    If the precision is not improved after a defined time interval, the learning rate
    will be multiplied by a decaying factor of 0.9"""

    def __init__(self, *args, **kwargs):
        super(PrecisionDecaySetter, self).__init__(*args, **kwargs)

        self.decay_refractory_steps = self.prm.train.train_control.learning_rate_setter.DECAY_REFRACTORY_STEPS
        if self.decay_refractory_steps is None:
            self.log.warning('DECAY_REFRACTORY_STEPS is None. setting it to be EVAL_STEPS * PRECISION_RETENTION_SIZE ({} x {})' \
                             .format(self.prm.train.train_control.EVAL_STEPS, self.prm.train.train_control.PRECISION_RETENTION_SIZE))
            self.decay_refractory_steps = self.prm.train.train_control.EVAL_STEPS * \
                                          self.prm.train.train_control.PRECISION_RETENTION_SIZE

        self.train_step_of_last_decay = 0

    def print_stats(self):
        super(PrecisionDecaySetter, self).print_stats()
        self.log.info(' DECAY_REFRACTORY_STEPS: {}'.format(self.decay_refractory_steps))

    def after_run(self, run_context, run_values):
        train_step = run_values.results
        if train_step - self.train_step_of_last_decay < self.decay_refractory_steps:
            # if we didn't wait decay_refractory_steps number of steps from the last decay, do nothing
            return
        if self.precision_retention.is_precision_stuck():
            self.log.info('train_step={}: Validtion precision did not improve after {} steps. decreasing the learning rate by a factor of {} to {}' \
                          .format(train_step, train_step - self.train_step_of_last_decay, 0.9, 0.9 * self._lrn_rate))
            self._lrn_rate *= 0.9
            self.train_step_of_last_decay = train_step
