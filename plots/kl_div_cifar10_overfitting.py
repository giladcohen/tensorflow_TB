from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(5.0, 8.0))

# wrn, cifar-10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/overfitting/log_0311_270918_metrics-SUPERSEED=27091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax1 = fig.add_subplot(211)
ax1.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax1.set_ylim(bottom=0, top=110)
ax1.set_title('CIFAR-10 overfitting')
ax1.yaxis.grid()
ax1.set_ylim(60, 70)
ax1.set_xticks([0, 5000, 10000, 15000, 20000])
ax1.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})

ax2 = fig.add_subplot(212)
ax2.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'k')
ax2.plot(data['test']['regular']['knn_kl_div2_avg']['steps'] , [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']] , 'k--')
# ax2.set_ylim(top=30)
ax2.set_xticks([0, 5000, 10000, 15000, 20000])
ax2.set_ylabel('KL divergence ($k$-NN||DNN)', labelpad=0.5, fontdict={'fontsize': 12})
ax2.yaxis.grid()
ax2.legend(['train', 'test'], loc=(0.77, 0.13))

plt.savefig('kl_div_cifar10_overfitting.png')