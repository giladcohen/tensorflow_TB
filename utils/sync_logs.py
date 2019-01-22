import subprocess
import time
import os

def run_cmd(cmd):
    print ('start running command:\n{}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command: {}'.format(cmd))


ip = str('35.241.190.74')
logdir_vec = []
for i in range(1, 9):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=21011900'.format(i))

# create all dictionaries
for logdir in logdir_vec:
    data_path = os.path.join(logdir, 'data_for_figures')
    if not os.path.exists(os.path.join(logdir, 'data_for_figures')):
        os.makedirs(data_path)
    data_path = os.path.join(logdir, 'test')
    if not os.path.exists(os.path.join(logdir, 'test')):
        os.makedirs(data_path)

    # cmd = 'scp -r gilad@132.66.196.128:' + logdir + '/data_for_figures ' + logdir + '/data_for_figures' # takes too long
    cmd = 'scp -r giladdalig@{}:'.format(ip) + logdir + '/data_for_figures/data.json ' + logdir + '/data_for_figures/data.json'
    run_cmd(cmd)
    # cmd = 'scp -r giladdalig@{}:'.format(ip) + logdir + '/train_validation_info.csv ' + logdir + '/train_validation_info.csv'
    # run_cmd(cmd)
    cmd = 'scp -r giladdalig@{}:'.format(ip) + logdir + '/test/train_features.npy ' + logdir + '/test/train_features.npy'
    run_cmd(cmd)
    cmd = 'scp -r giladdalig@{}:'.format(ip) + logdir + '/test/test_features.npy ' + logdir + '/test/test_features.npy'
    run_cmd(cmd)
    cmd = 'scp -r giladdalig@{}:'.format(ip) + logdir + '/test/train_labels.npy ' + logdir + '/test/train_labels.npy'
    run_cmd(cmd)
    cmd = 'scp -r giladdalig@{}:'.format(ip) + logdir + '/test/test_labels.npy ' + logdir + '/test/test_labels.npy'
    run_cmd(cmd)
    cmd = 'scp -r giladdalig@{}:'.format(ip) + logdir + '/test/train_dnn_predictions_prob.npy ' + logdir + '/test/train_dnn_predictions_prob.npy'
    run_cmd(cmd)
    cmd = 'scp -r giladdalig@{}:'.format(ip) + logdir + '/test/test_dnn_predictions_prob.npy ' + logdir + '/test/test_dnn_predictions_prob.npy'
    run_cmd(cmd)
    # cmd += 'scp -r ' + logdir + ' gilad@{}:'.format(ip) + logdir
    # run_cmd(cmd)


print('end of script.')
