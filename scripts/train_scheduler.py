import subprocess
import time

logdir_vec = [
    '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111805',
    '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111806',
    '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111807',
    '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111808',
    '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111809',
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
          ' -c examples/train/train_multi_sf_mnist.ini'
    run_cmd(cmd)


logdir_vec = [
    '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111805',
    '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111806',
    '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111807',
    '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111808',
    '/data/gilad/logs/multi_sf/mnist/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111809',
]

for logdir in logdir_vec:
    cmd = 'CUDA_VISIBLE_DEVICES=3 python scripts/train_automated.py' + \
          ' --ROOT_DIR ' + logdir + \
          ' --SUPERSEED ' + logdir[-8:] + \
          ' --MULTI_SF True' + \
          ' -c examples/train/train_multi_sf_mnist.ini'
    run_cmd(cmd)

print('end of script.')
