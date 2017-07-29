from lib.trainers.hooks.learning_rate_setter_base import LearningRateSetterBase

class FixedScheduleSetter(LearningRateSetterBase):
    """Decreasing the learning rate in fixed steps, using fixed decay rate"""

    def __init__(self, *args, **kwargs):
        super(FixedScheduleSetter, self).__init__(*args, **kwargs)

        self.scheduled_epochs        = self.prm.train.train_control.learning_rate_setter.SCHEDULED_EPOCHS
        self.scheduled_reaning_rates = self.prm.train.train_control.learning_rate_setter.SCHEDULED_LEARNING_RATES
        self.all_learning_rates      = [self._init_lrn_rate] + self.scheduled_reaning_rates # all learning rates

        self.train_batch_size = self.trainset_dataset.batch_size
        self.train_set_size   = self.trainset_dataset.pool_size()
        self._notify = [False] * len(self.all_learning_rates)

        self.assert_config()

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        epoch = (self.train_batch_size * global_step) // self.train_set_size
        for i in range(len(self.scheduled_epochs)):
            if epoch < self.scheduled_epochs[i]:
                if not self._notify[i]:
                    self.log.info('epoch={}. setting learning rate to {}'.format(epoch, self.all_learning_rates[i]))
                    self.set_lrn_rate(self.all_learning_rates[i])
                    self._notify[i] = True
                return
        if not self._notify[i+1]:
            self.log.info('epoch={}. setting learning rate to {}'.format(epoch, self.all_learning_rates[i+1]))
            self.set_lrn_rate(self.all_learning_rates[i+1])
            self._notify[i+1] = True

    def print_stats(self):
        super(FixedScheduleSetter, self).print_stats()
        self.log.info(' SCHEDULED_EPOCHS: {}'.format(self.scheduled_epochs))
        self.log.info(' SCHEDULED_LEARNING_RATES: {}'.format(self.scheduled_reaning_rates))
        self.log.info(' ALL_LEARNING_RATES: {}'.format(self.all_learning_rates))
        self.log.info(' [DEBUG]: Using in hook: train_batch_size={}, train_set_size={}'.format(self.train_batch_size, self.train_set_size))

    def assert_config(self):
        if len(self.scheduled_epochs) != len(self.scheduled_reaning_rates):
            err_str = 'SCHEDULED_EPOCHS ({}) and SCHEDULED_LEARNING_RATES ({}) must have the same lengths'.format(self.scheduled_epochs, self.scheduled_reaning_rates)
            self.log.error(err_str)
            raise AssertionError(err_str)
