"""In this script I print:
the 1-NN error rate as a function of the train samples
 +
the 1-NN error rate - DNN error rate as a function of the train samples
 """

from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
subpos = np.array([0.35, 0.25, 0.5, 0.4])
fig = plt.figure(figsize=(15.0, 8.0))

measure = 'dnn_score'
# setting all experiments
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
n_vec = [30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900]
for i in range(1, 61):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=23111800'.format(i))
    n_vec.append(int(i * 1000))

dnn_error_rate = []
for root_dir in logdir_vec:
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    with open(json_file) as f:
        data = json.load(f)
    dnn_error_rate.append(1.0 - data['test']['regular'][measure]['values'][0])

ax1 = fig.add_subplot(111)
ax1.semilogx(n_vec, dnn_error_rate)
ax1.yaxis.grid()
ax1.set_ylabel('DNN error rate', labelpad=5, fontdict={'fontsize': 12})
ax1.set_xlabel('number of samples')
ax1.set_title('DNN error rate')

plt.tight_layout()
plt.savefig('dnn_error_rate.png')