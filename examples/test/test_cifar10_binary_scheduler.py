import subprocess
import time
import numpy as np
import os

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))
    time.sleep(3)

# knn_norm = 'L1'
# n_vec = np.array([8])

logdir_vec = [
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_7k-SUPERSEED=21011903',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_8k-SUPERSEED=21011901',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_9k-SUPERSEED=21011901',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_10k-SUPERSEED=21011901'
]

# for n in n_vec:
for logdir in logdir_vec:
    train_validation_info = os.path.join(logdir, 'train_validation_info.csv')
    cmd = 'CUDA_VISIBLE_DEVICES=3 python scripts/test_automated.py' + \
          ' --ROOT_DIR ' + logdir + \
          ' --KNN_NORM ' + 'L1' + \
          ' --PCA_REDUCTION False' + \
          ' --CHECKPOINT_FILE ' + 'model_schedule.ckpt-25000' + \
          ' --TRAIN_VALIDATION_MAP_REF ' + train_validation_info + \
          ' --DROPOUT_KEEP_PROB 1.0' + \
          ' -c examples/test/test_multi_knn.ini'
    run_cmd(cmd)
    cmd = 'CUDA_VISIBLE_DEVICES=3 python scripts/test_automated.py' + \
          ' --ROOT_DIR ' + logdir + \
          ' --KNN_NORM ' + 'L2' + \
          ' --PCA_REDUCTION False' + \
          ' --CHECKPOINT_FILE ' + 'model_schedule.ckpt-25000' + \
          ' --TRAIN_VALIDATION_MAP_REF ' + train_validation_info + \
          ' --DROPOUT_KEEP_PROB 1.0' + \
          ' --DUMP_NET True' + \
          ' -c examples/test/test_multi_knn.ini'
    run_cmd(cmd)

print('end of script.')
