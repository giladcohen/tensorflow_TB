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
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_1k-SUPERSEED=08011900',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_2k-SUPERSEED=08011900',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_3k-SUPERSEED=08011900',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_4k-SUPERSEED=08011900',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_5k-SUPERSEED=08011900',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_6k-SUPERSEED=08011900',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_7k-SUPERSEED=08011900',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_8k-SUPERSEED=08011900',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_9k-SUPERSEED=08011900',
]

# create all dictionaries
for logdir in logdir_vec:
    data_path = os.path.join(logdir, 'data_for_figures')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # cmd = 'scp -r gilad@132.66.196.128:' + logdir + '/data_for_figures ' + logdir + '/data_for_figures' # takes too long
    cmd = 'scp -r gilad@132.66.196.128:' + logdir + '/data_for_figures/data.json ' + logdir + '/data_for_figures/data.json'
    run_cmd(cmd)

print('end of script.')
