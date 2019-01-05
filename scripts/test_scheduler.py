import subprocess
import time
import os

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))
    time.sleep(3)

# logdir_vec = [
#     '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.2k-SUPERSEED=19121800',
#     '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.3k-SUPERSEED=19121800',
#     '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.4k-SUPERSEED=19121800',
#     '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.5k-SUPERSEED=19121800',
#     '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.6k-SUPERSEED=19121800',
#     '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.7k-SUPERSEED=19121800',
#     '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.8k-SUPERSEED=19121800',
#     '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.9k-SUPERSEED=19121800',
# ]

# num_of_iters_vec = [20000, 15000, 12000, 10000, 8571, 7500, 6667, 6000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]


logdir_vec = []
for i in range(1, 17):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/cifar10/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=30121800'.format(i))

knn_norm = 'L1'

for i, logdir in enumerate(logdir_vec):
    train_validation_info = os.path.join(logdir, 'train_validation_info.csv')
    cmd = 'CUDA_VISIBLE_DEVICES=0 python scripts/test_automated.py' + \
          ' --ROOT_DIR ' + logdir + \
          ' --KNN_NORM ' + knn_norm + \
          ' --PCA_REDUCTION False' + \
          ' --CHECKPOINT_FILE ' + 'model_schedule.ckpt-50000' + \
          ' --TRAIN_VALIDATION_MAP_REF ' + train_validation_info + \
          ' -c examples/test/test_multi_knn.ini'
    run_cmd(cmd)

print('end of script.')
