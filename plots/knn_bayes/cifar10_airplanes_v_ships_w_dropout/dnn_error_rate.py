"""In this script I print:
the 1-NN error rate as a function of the train samples
 +
the 1-NN error rate - DNN error rate as a function of the train samples
 """

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

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
logdir_vec = []
n_vec      = []
for i in range(1, 11):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=21011900'.format(i))
    n_vec.append(int(i * 1000))

dnn_error_rate = []
for root_dir in logdir_vec:
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    with open(json_file) as f:
        data = json.load(f)
    dnn_error_rate.append(1.0 - data['test']['regular'][measure]['values'][0])

dnn_error_rate = np.array(dnn_error_rate)
approx_bayes_error_rate = 0.0047

ax1 = fig.add_subplot(111)
ax1.plot(n_vec, dnn_error_rate - approx_bayes_error_rate)
ax1.yaxis.grid()
ax1.set_ylabel('DNN error rate', labelpad=5, fontdict={'fontsize': 12})
ax1.set_xlabel('number of samples')
ax1.set_title('CIFAR10_cars_v_trucks DNN error rate')

plt.tight_layout()
plt.savefig('dnn_error_rate.png')