from tensorflow_TB.utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(15.0, 6.0))

# wrn, mnist
root_dir = '/data/gilad/logs/metrics/wrn/mnist/overfitting/log_2046_061018_metrics-SUPERSEED=06101800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax1 = fig.add_subplot(231)
ax1.plot(data['test']['regular']['dnn_score']['steps'][:110], [v * 100 for v in data['test']['regular']['dnn_score']['values'][:110]], 'k')
ax1.set_title('MNIST')
ax1.yaxis.grid()
ax1.set_ylim(60, 95)
ax1.set_xticks([0, 500, 1000, 1500, 2000])
ax1.set_ylabel('accuracy (%)', labelpad=10, fontdict={'fontsize': 12})

ax2 = fig.add_subplot(234)
ax2.plot(data['train']['regular']['knn_kl_div2_avg']['steps'][:110], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values'][:110]], 'k')
ax2.plot(data['test']['regular']['knn_kl_div2_avg']['steps'][:110] , [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values'][:110]] , 'k--')
# ax2.set_ylim(top=30)
ax2.set_xticks([0, 500, 1000, 1500, 2000])
ax2.set_ylabel('$D_{KL}$($k$-NN || DNN)', labelpad=2, fontdict={'fontsize': 12})
ax2.yaxis.grid()
ax2.legend(['train', 'test'], loc=(0.75, 0.15))

# # wrn, cifar-10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/overfitting/log_0311_270918_metrics-SUPERSEED=27091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax3 = fig.add_subplot(232)
ax3.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax3.set_ylim(bottom=0, top=110)
ax3.set_title('CIFAR-10')
ax3.yaxis.grid()
ax3.set_ylim(60, 70)
ax3.set_xticks([0, 5000, 10000, 15000, 20000])

ax4 = fig.add_subplot(235)
ax4.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'k')
ax4.plot(data['test']['regular']['knn_kl_div2_avg']['steps'] , [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']] , 'k--')
# ax4.set_ylim(top=30)
ax4.set_xticks([0, 5000, 10000, 15000, 20000])
ax4.yaxis.grid()
ax4.legend(['train', 'test'], loc=(0.77, 0.19))

# # wrn, cifar-100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/overfitting/log_2046_061018_metrics-SUPERSEED=06101803'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax5 = fig.add_subplot(233)
ax5.plot(data['test']['regular']['dnn_score']['steps'][:74], [v * 100 for v in data['test']['regular']['dnn_score']['values'][0:74]], 'k')
ax5.set_ylim(bottom=10, top=33)
ax5.set_title('CIFAR-100')
ax5.yaxis.grid()
# ax5.set_ylim(60, 70)
ax5.set_xticks([0, 5000, 10000, 15000])

ax6 = fig.add_subplot(236)
ax6.plot(data['train']['regular']['knn_kl_div2_avg']['steps'][:74], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values'][:74]], 'k')
ax6.plot(data['test']['regular']['knn_kl_div2_avg']['steps'][:74] , [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values'][:74]] , 'k--')
# ax4.set_ylim(top=30)
ax6.set_xticks([0, 5000, 10000, 15000])
ax6.yaxis.grid()
ax6.legend(['train', 'test'], loc=(0.77, 0.13))



plt.tight_layout()
plt.savefig('kl_div_overfitting.png')