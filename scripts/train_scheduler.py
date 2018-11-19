import subprocess
import time

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
]

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))
    time.sleep(3)

for logdir in logdir_vec:
    cmd = 'CUDA_VISIBLE_DEVICES=3 python scripts/train_automated.py' + \
          ' --ROOT_DIR ' + logdir + \
          ' --SUPERSEED ' + logdir[-8:] + \
          ' --MULTI_SF False' + \
          ' --ARCHITECTURE ' + 'Wide-Resnet-28-10' \
          ' -c examples/train/train_multi_sf_mnist.ini'
    run_cmd(cmd)


logdir_vec = [
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
]

for logdir in logdir_vec:
    cmd = 'CUDA_VISIBLE_DEVICES=3 python scripts/train_automated.py' + \
          ' --ROOT_DIR ' + logdir + \
          ' --SUPERSEED ' + logdir[-8:] + \
          ' --MULTI_SF True' + \
          ' --ARCHITECTURE ' + 'Wide-Resnet-28-10_MultiSf' \
          ' -c examples/train/train_multi_sf_mnist.ini'
    run_cmd(cmd)

print('end of script.')
