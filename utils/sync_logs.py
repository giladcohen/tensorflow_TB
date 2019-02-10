import subprocess
import time
import os

def run_cmd(cmd):
    print ('start running command:\n{}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command: {}'.format(cmd))


ip  = '132.66.196.128'
usr = 'gilad'
logdir_vec = [
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_7k-SUPERSEED=21011903',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_8k-SUPERSEED=21011901',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_9k-SUPERSEED=21011901',
    '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_10k-SUPERSEED=21011901'
]

# create all dictionaries
for logdir in logdir_vec:
    data_path = os.path.join(logdir, 'data_for_figures')
    if not os.path.exists(os.path.join(logdir, 'data_for_figures')):
        os.makedirs(data_path)
    data_path = os.path.join(logdir, 'test')
    if not os.path.exists(os.path.join(logdir, 'test')):
        os.makedirs(data_path)

    # cmd = 'scp -r {}@{}:'.format(usr, ip) + logdir + '/data_for_figures ' + logdir + '/data_for_figures' # takes too long
    cmd = 'scp -r {}@{}:'.format(usr, ip) + logdir + '/data_for_figures/data.json ' + logdir + '/data_for_figures/data.json'
    run_cmd(cmd)
    # cmd = 'scp -r {}@{}:'.format(usr, ip) + logdir + '/train_validation_info.csv ' + logdir + '/train_validation_info.csv'
    # run_cmd(cmd)
    cmd = 'scp -r {}@{}:'.format(usr, ip) + logdir + '/test/train_features.npy ' + logdir + '/test/train_features.npy'
    run_cmd(cmd)
    cmd = 'scp -r {}@{}:'.format(usr, ip) + logdir + '/test/test_features.npy ' + logdir + '/test/test_features.npy'
    run_cmd(cmd)
    cmd = 'scp -r {}@{}:'.format(usr, ip) + logdir + '/test/train_labels.npy ' + logdir + '/test/train_labels.npy'
    run_cmd(cmd)
    cmd = 'scp -r {}@{}:'.format(usr, ip) + logdir + '/test/test_labels.npy ' + logdir + '/test/test_labels.npy'
    run_cmd(cmd)
    cmd = 'scp -r {}@{}:'.format(usr, ip) + logdir + '/test/train_dnn_predictions_prob.npy ' + logdir + '/test/train_dnn_predictions_prob.npy'
    run_cmd(cmd)
    cmd = 'scp -r {}@{}:'.format(usr, ip) + logdir + '/test/test_dnn_predictions_prob.npy ' + logdir + '/test/test_dnn_predictions_prob.npy'
    run_cmd(cmd)
    # cmd += 'scp -r ' + logdir + ' gilad@{}:'.format(ip) + logdir
    # run_cmd(cmd)


print('end of script.')
