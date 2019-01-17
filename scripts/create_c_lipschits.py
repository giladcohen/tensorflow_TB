import subprocess
import time
import os

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))
    time.sleep(3)

norm_vec = ['L1', 'L2']
knn_norm = 'L1'

for i, logdir in enumerate(logdir_vec):
    train_validation_info = os.path.join(logdir, 'train_validation_info.csv')
    cmd = 'CUDA_VISIBLE_DEVICES=0 python scripts/test_automated.py' + \
          ' --ROOT_DIR ' + logdir + \
          ' --KNN_NORM ' + knn_norm + \
          ' --PCA_REDUCTION False' + \
          ' --CHECKPOINT_FILE ' + 'model_schedule.ckpt-50000' + \
          ' --TRAIN_VALIDATION_MAP_REF ' + train_validation_info + \
          ' --DROPOUT_KEEP_PROB 1.0' + \
          ' -c examples/test/test_bayesian_multi_knn.ini'
    run_cmd(cmd)

print('end of script.')
