import subprocess
import time
import os

logdir_vec   = [
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111800',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111801',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111802',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111803',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111804',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111805',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111806',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111807',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111808',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111809',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111800',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111801',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111802',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111803',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111804',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111805',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111806',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111807',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111808',
        '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111809',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111800',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111801',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111802',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111803',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111804',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111805',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111806',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111807',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111808',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111809',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111800',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111801',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111802',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111803',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111804',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111805',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111806',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111807',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111808',
        '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111809',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111800',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111801',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111802',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111803',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111804',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111805',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111806',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111807',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111808',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111809',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111800',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111801',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111802',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111803',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111804',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111805',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111806',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111807',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111808',
        '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111809',
    ]

def run_cmd(cmd):
    print ('start running command:\n{}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command: {}'.format(cmd))
    time.sleep(3)


# create all dictionaries
for logdir in logdir_vec:
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    cmd = 'scp -r gilad@132.66.196.128:' + logdir + '/data_for_figures ' + logdir + '/data_for_figures'
    run_cmd(cmd)

print('end of script.')















