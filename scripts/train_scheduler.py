import subprocess
import time

logdir_vec = [
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111805',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111806',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111807',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111808',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111809',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111810',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111811',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111812',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111813',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111814',
]

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))
    time.sleep(3)

for logdir in logdir_vec:
    # create dump once:
    cmd = 'CUDA_VISIBLE_DEVICES=0 python scripts/train_automated.py' + \
          ' --ROOT_DIR ' + logdir + \
          ' --SUPERSEED ' + logdir[-8:] + \
          ' -c examples/train/train_multi_sf_cifar10.ini'
    run_cmd(cmd)

print('end of script.')
