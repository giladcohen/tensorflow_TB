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

measure = 'knn_k_1_norm_L1_knn_score'
# setting all experiments
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
n_vec = [200, 300, 400, 500, 600, 700, 800, 900]

for i in range(1, 51):
    if i in [4, 13]:  # not ready yet
        continue
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/cifar10/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=19121800'.format(i))
    n_vec.append(int(i * 1000))

knn_error_rate = []
for root_dir in logdir_vec:
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    with open(json_file) as f:
        data = json.load(f)
    knn_error_rate.append(1.0 - data['test']['regular'][measure]['values'][0])

ax1 = fig.add_subplot(111)
ax1.semilogx(n_vec, knn_error_rate)
ax1.yaxis.grid()
ax1.set_ylabel('1-NN error rate', labelpad=5, fontdict={'fontsize': 12})
ax1.set_xlabel('number of samples')
ax1.set_title('KNN error rate (K=1)')

plt.tight_layout()
plt.savefig('1_nn_error_rate.png')