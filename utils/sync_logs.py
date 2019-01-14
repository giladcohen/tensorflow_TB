import subprocess
import time
import os

def run_cmd(cmd):
    print ('start running command:\n{}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command: {}'.format(cmd))


# logdir_vec = []
# for i in range(1, 17):
#     logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/mnist/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=30121800'.format(i))

logdir_vec = [
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_10k-SUPERSEED=08011900',
]

# create all dictionaries
for logdir in logdir_vec:
    data_path = os.path.join(logdir, 'data_for_figures')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # cmd = 'scp -r gilad@132.66.196.128:' + logdir + '/data_for_figures ' + logdir + '/data_for_figures' # takes too long
    cmd = 'scp -r giladdalig@35.233.9.152:' + logdir + '/data_for_figures/data.json ' + logdir + '/data_for_figures/data.json'
    run_cmd(cmd)
    cmd = 'scp -r giladdalig@35.233.9.152:' + logdir + '/train_validation_info.csv ' + logdir + '/train_validation_info.csv'
    run_cmd(cmd)
    cmd = 'scp -r giladdalig@35.233.9.152:' + logdir + '/test/train_features.npy ' + logdir + '/test/train_features.npy'
    run_cmd(cmd)
    cmd = 'scp -r giladdalig@35.233.9.152:' + logdir + '/test/test_features.npy ' + logdir + '/test/test_features.npy'
    run_cmd(cmd)
    cmd = 'scp -r giladdalig@35.233.9.152:' + logdir + '/test/train_labels.npy ' + logdir + '/test/train_labels.npy'
    run_cmd(cmd)
    cmd = 'scp -r giladdalig@35.233.9.152:' + logdir + '/test/test_labels.npy ' + logdir + '/test/test_labels.npy'
    run_cmd(cmd)

print('end of script.')
