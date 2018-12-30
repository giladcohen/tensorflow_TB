from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
subpos = np.array([0.35, 0.25, 0.5, 0.4])
fig = plt.figure(figsize=(15.0, 8.0))

# setting all experiments
all_ks = [1, 3, 4, 5, 6, 7, 8, 9, 10,
          12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
          45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
          110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
          220, 240, 260, 280, 300,
          350, 400, 450, 500,
          600, 700, 800, 900, 1000]

ploted_ks = [1, 3, 4, 5, 8, 18, 26, 40, 70, 100, 200, 300, 500, 700, 1000]

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
n_vec  = [30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900]
max_ks = [3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 20 , 30 , 40 , 50 , 60 , 70 , 80 , 90]
for i in range(1, 61):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/mnist/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=23111800'.format(i))
    n_vec.append(int(i * 1000))
    max_ks.append(int(i * 100))

optimal_knn_error_rate = []
optimal_k              = []
knn_error_rate         = {}
for i, root_dir in enumerate(logdir_vec):
    n = n_vec[i]
    knn_error_rate[n] = {}  # every k in an element here
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    max_k = max_ks[i]
    with open(json_file) as f:
        data = json.load(f)
    best_error_rate = np.inf
    best_k          = None
    for k in all_ks:
        if k <= max_k:
            measure = 'knn_k_{}_norm_L2_knn_kl_div2_median'.format(k)
            # knn_error_rate[n][k] = 1.0 - data['test']['regular'][measure]['values'][0]
            knn_error_rate[n][k] = data['test']['regular'][measure]['values'][0]
            if knn_error_rate[n][k] < best_error_rate:
                best_error_rate = knn_error_rate[n][k]
                best_k = k
    optimal_knn_error_rate.append(best_error_rate)
    optimal_k.append(best_k)

ax = fig.add_subplot(111)
for i, k in enumerate(ploted_ks):
    n_vals   = []
    err_vals = []
    for n in n_vec:
        if k in sorted(knn_error_rate[n].keys()):
            n_vals.append(n)
            err_vals.append(knn_error_rate[n][k])

    ax.plot(n_vals, err_vals, label='k={}'.format(k))
    ax.yaxis.grid()
    ax.set_ylabel('k-NN error rate', labelpad=5, fontdict={'fontsize': 12})
    ax.set_xlabel('number of samples')
    ax.set_title('KNN error rate for many ks')
    ax.set_xlim(left=4000, right=6000)
    ax.set_ylim(bottom=0.043, top=0.045)

plt.legend()
plt.tight_layout()
plt.savefig('knn_error_rate_many_ks.png')

