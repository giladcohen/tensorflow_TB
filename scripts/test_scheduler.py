import subprocess
import time
import os

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))
    time.sleep(3)

logdir_vec = []
for train_size in range(1, 61):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=23111800'.format(train_size))

knn_norm = 'L1'

for logdir in logdir_vec:
    train_validation_info = os.path.join(logdir, 'train_validation_info.csv')
    cmd = 'CUDA_VISIBLE_DEVICES=2 python scripts/test_automated.py' + \
          ' --ROOT_DIR ' + logdir + \
          ' --KNN_NORM ' + knn_norm + \
          ' --CHECKPOINT_FILE ' + 'model_schedule.ckpt-3000' \
          ' --TRAIN_VALIDATION_MAP_REF ' + train_validation_info + \
          ' -c examples/test/test_multi_knn.ini'
    run_cmd(cmd)

print('end of script.')
