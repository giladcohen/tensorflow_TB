import subprocess
import time
import os

def run_cmd(cmd):
    print ('start running command:\n{}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command: {}'.format(cmd))


logdir_vec = [
    '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.2k-SUPERSEED=19121800',
    '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.3k-SUPERSEED=19121800',
    '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.4k-SUPERSEED=19121800',
    '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.5k-SUPERSEED=19121800',
    '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.6k-SUPERSEED=19121800',
    '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.7k-SUPERSEED=19121800',
    '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.8k-SUPERSEED=19121800',
    '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_0.9k-SUPERSEED=19121800',
]

for i in range(1, 51):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=19121800'.format(i))

# create all dictionaries
for logdir in logdir_vec:
    if logdir in [
        '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_4k-SUPERSEED=19121800',
        '/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_13k-SUPERSEED=19121800',
    ]:
        continue
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    cmd = 'scp -r gilad@132.66.196.128:' + logdir + '/data_for_figures ' + logdir + '/data_for_figures'
    run_cmd(cmd)

print('end of script.')





Fuad2103






