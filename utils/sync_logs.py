import subprocess
import time
import os

logdir_vec = [
        '/data/gilad/logs/multi_sf/cifar10/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111800',
        '/data/gilad/logs/multi_sf/cifar10/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111801',
        '/data/gilad/logs/multi_sf/cifar10/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111802',
        '/data/gilad/logs/multi_sf/cifar10/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111803',
        '/data/gilad/logs/multi_sf/cifar10/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111804',
        '/data/gilad/logs/multi_sf/cifar10/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111805',
        '/data/gilad/logs/multi_sf/cifar10/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111806',
        '/data/gilad/logs/multi_sf/cifar10/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111807',
        '/data/gilad/logs/multi_sf/cifar10/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111808',
        '/data/gilad/logs/multi_sf/cifar10/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111809',
        '/data/gilad/logs/multi_sf/cifar10/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111800',
        '/data/gilad/logs/multi_sf/cifar10/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111801',
        '/data/gilad/logs/multi_sf/cifar10/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111802',
        '/data/gilad/logs/multi_sf/cifar10/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111803',
        '/data/gilad/logs/multi_sf/cifar10/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111804',
        '/data/gilad/logs/multi_sf/cifar10/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111805',
        '/data/gilad/logs/multi_sf/cifar10/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111806',
        '/data/gilad/logs/multi_sf/cifar10/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111807',
        '/data/gilad/logs/multi_sf/cifar10/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111808',
        '/data/gilad/logs/multi_sf/cifar10/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111809',
        '/data/gilad/logs/multi_sf/cifar100/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111800',
        '/data/gilad/logs/multi_sf/cifar100/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111801',
        '/data/gilad/logs/multi_sf/cifar100/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111802',
        '/data/gilad/logs/multi_sf/cifar100/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111803',
        '/data/gilad/logs/multi_sf/cifar100/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111804',
        '/data/gilad/logs/multi_sf/cifar100/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111805',
        # '/data/gilad/logs/multi_sf/cifar100/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111806',
        # '/data/gilad/logs/multi_sf/cifar100/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111807',
        # '/data/gilad/logs/multi_sf/cifar100/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111808',
        # '/data/gilad/logs/multi_sf/cifar100/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111809',
        '/data/gilad/logs/multi_sf/cifar100/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111800',
        '/data/gilad/logs/multi_sf/cifar100/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111801',
        '/data/gilad/logs/multi_sf/cifar100/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111802',
        '/data/gilad/logs/multi_sf/cifar100/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111803',
        '/data/gilad/logs/multi_sf/cifar100/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111804',
        '/data/gilad/logs/multi_sf/cifar100/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111805',
        # '/data/gilad/logs/multi_sf/cifar100/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111806',
        # '/data/gilad/logs/multi_sf/cifar100/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111807',
        # '/data/gilad/logs/multi_sf/cifar100/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111808',
        # '/data/gilad/logs/multi_sf/cifar100/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111809',
        '/data/gilad/logs/multi_sf/mnist/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111800',
        '/data/gilad/logs/multi_sf/mnist/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111801',
        '/data/gilad/logs/multi_sf/mnist/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111802',
        '/data/gilad/logs/multi_sf/mnist/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111803',
        # '/data/gilad/logs/multi_sf/mnist/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111804',
        # '/data/gilad/logs/multi_sf/mnist/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111805',
        # '/data/gilad/logs/multi_sf/mnist/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111806',
        # '/data/gilad/logs/multi_sf/mnist/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111807',
        # '/data/gilad/logs/multi_sf/mnist/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111808',
        # '/data/gilad/logs/multi_sf/mnist/small/log_no_multi_sf_lr_0.01-SUPERSEED=19111809',
        '/data/gilad/logs/multi_sf/mnist/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111800',
        '/data/gilad/logs/multi_sf/mnist/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111801',
        '/data/gilad/logs/multi_sf/mnist/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111802',
        '/data/gilad/logs/multi_sf/mnist/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111803',
        # '/data/gilad/logs/multi_sf/mnist/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111804',
        # '/data/gilad/logs/multi_sf/mnist/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111805',
        # '/data/gilad/logs/multi_sf/mnist/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111806',
        # '/data/gilad/logs/multi_sf/mnist/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111807',
        # '/data/gilad/logs/multi_sf/mnist/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111808',
        # '/data/gilad/logs/multi_sf/mnist/small/log_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=19111809',
]


def run_cmd(cmd):
    print ('start running command:\n{}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command: {}'.format(cmd))

# create all dictionaries
for logdir in logdir_vec:
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    cmd = 'scp -r gilad@132.66.196.128:' + logdir + '/data_for_figures ' + logdir + '/data_for_figures'
    run_cmd(cmd)

print('end of script.')















