import subprocess
import time
import numpy as np

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))
    time.sleep(3)

n_vec = np.array([1, 2, 3])

for n in n_vec:
    logdir = '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=08011900'.format(n)
    cmd = 'CUDA_VISIBLE_DEVICES=0 python scripts/train_automated.py' + \
          ' --ROOT_DIR ' + logdir + \
          ' --SUPERSEED ' + logdir[-8:] + \
          ' --TRAIN_SET_SIZE ' + str(int(n * 1000)) + \
          ' --ARCHITECTURE ' + 'Wide-Resnet-28-10' \
          ' --DROPOUT_KEEP_PROB 0.5' + \
          ' -c examples/train/train_cifar10_cats_v_dogs_50k_iters.ini'
    run_cmd(cmd)

print('end of script.')
