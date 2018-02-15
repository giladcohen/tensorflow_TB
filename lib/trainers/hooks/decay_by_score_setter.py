from lib.trainers.hooks.learning_rate_setter_base import LearningRateSetterBase

class DecayByScoreSetter(LearningRateSetterBase):
    """Decaying the learning rate based on the score retention.
    If the score is not improved after a defined time interval, the learning rate
    will be multiplied by a decaying factor of 0.9, as done in:
    https://arxiv.org/abs/1705.08292
    """

    def __init__(self, *args, **kwargs):
        super(DecayByScoreSetter, self).__init__(*args, **kwargs)

        self.decay_refractory_steps    = self.prm.train.train_control.learning_rate_setter.LR_DECAY_REFRACTORY_STEPS
        self.global_step_of_last_decay = 0

    def print_stats(self):
        super(DecayByScoreSetter, self).print_stats()
        self.log.info(' LR_DECAY_REFRACTORY_STEPS: {}'.format(self.decay_refractory_steps))

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        if global_step - self.global_step_of_last_decay < self.decay_refractory_steps:
            # if we didn't wait decay_refractory_steps number of steps from the last decay, do nothing
            return
        if self.retention.is_score_stuck():
            self.log.info('global_step={}: Validtion score did not improve after {} steps. Decreasing the learning rate by a factor of {} to {}. Last learning rate decay was before {} steps.' \
                          .format(global_step, global_step - self.retention.best_score_step, 0.9, 0.9 * self._lrn_rate, global_step - self.global_step_of_last_decay))
            self.set_lrn_rate(0.9 * self._lrn_rate)
            self.global_step_of_last_decay = global_step
