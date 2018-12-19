import subprocess
import time
import os

def run_cmd(cmd):
    print ('start running command:\n{}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command: {}'.format(cmd))


logdir_vec = [
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_30_lr_0.1s_n_0.03k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_40_lr_0.1s_n_0.04k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_50_lr_0.1s_n_0.05k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_60_lr_0.1s_n_0.06k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_70_lr_0.1s_n_0.07k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_80_lr_0.1s_n_0.08k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_90_lr_0.1s_n_0.09k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_100_lr_0.1s_n_0.1k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_200_lr_0.1s_n_0.2k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_200_lr_0.1s_n_0.3k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_200_lr_0.1s_n_0.4k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_200_lr_0.1s_n_0.5k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_200_lr_0.1s_n_0.6k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_200_lr_0.1s_n_0.7k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_200_lr_0.1s_n_0.8k-SUPERSEED=23111800',
    '/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_200_lr_0.1s_n_0.9k-SUPERSEED=23111800',
]

for i in range(1, 61):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=23111800'.format(i))

# create all dictionaries
for logdir in logdir_vec:
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    cmd = 'scp -r gilad@132.66.196.128:' + logdir + '/data_for_figures ' + logdir + '/data_for_figures'
    run_cmd(cmd)

print('end of script.')











