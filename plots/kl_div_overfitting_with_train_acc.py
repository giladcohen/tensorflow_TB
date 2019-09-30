from tensorflow_TB.utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(15.0, 8.0))

# wrn, mnist
root_dir = '/data/gilad/logs/metrics/wrn/mnist/overfitting/log_2046_061018_metrics-SUPERSEED=06101800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)

ax1 = fig.add_subplot(331)
ax1.plot(data['train']['regular']['dnn_score']['steps'][:110], [v * 100 for v in data['train']['regular']['dnn_score']['values'][:110]], 'k')
ax1.set_title('MNIST')
ax1.yaxis.grid()
ax1.set_ylim(0, 104)
ax1.set_xticks([0, 500, 1000, 1500, 2000])
ax1.set_ylabel('train accuracy (%)', labelpad=10, fontdict={'fontsize': 12})

ax2 = fig.add_subplot(334)
# ax2.plot(340, 84.25, 'ro')
ax2.plot(data['test']['regular']['dnn_score']['steps'][:110], [v * 100 for v in data['test']['regular']['dnn_score']['values'][:110]], 'k')
ax2.yaxis.grid()
ax2.set_ylim(60, 95)
ax2.set_xticks([0, 500, 1000, 1500, 2000])
ax2.set_ylabel('test accuracy (%)', labelpad=10, fontdict={'fontsize': 12})

ax3 = fig.add_subplot(337)
ax3.plot(data['train']['regular']['knn_kl_div2_avg']['steps'][:110], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values'][:110]], 'k')
ax3.plot(data['test']['regular']['knn_kl_div2_avg']['steps'][:110] , [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values'][:110]] , 'k--')
# ax3.set_ylim(top=30)
ax3.set_xticks([0, 500, 1000, 1500, 2000])
ax3.set_ylabel('$D_{KL}$($k$-NN || DNN)', labelpad=2, fontdict={'fontsize': 12})
ax3.yaxis.grid()
ax3.legend(['train', 'test'], loc=(0.75, 0.15))


# wrn, cifar-10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/overfitting/log_0311_270918_metrics-SUPERSEED=27091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)

ax4 = fig.add_subplot(332)
ax4.plot(data['train']['regular']['dnn_score']['steps'], [v * 100 for v in data['train']['regular']['dnn_score']['values']], 'k')
ax4.set_ylim(bottom=0, top=104)
ax4.set_title('CIFAR-10')
ax4.yaxis.grid()
ax4.set_xticks([0, 5000, 10000, 15000, 20000])

ax5 = fig.add_subplot(335)
# ax5.plot(1000, 67.34, 'ro')
ax5.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax5.set_ylim(bottom=0, top=110)
ax5.yaxis.grid()
ax5.set_ylim(60, 70)
ax5.set_xticks([0, 5000, 10000, 15000, 20000])

ax6 = fig.add_subplot(338)
ax6.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'k')
ax6.plot(data['test']['regular']['knn_kl_div2_avg']['steps'] , [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']] , 'k--')
# ax6.set_ylim(top=30)
ax6.set_xticks([0, 5000, 10000, 15000, 20000])
ax6.yaxis.grid()
ax6.legend(['train', 'test'], loc=(0.77, 0.19))


# wrn, cifar-100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/overfitting/log_2046_061018_metrics-SUPERSEED=06101803'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)

ax7 = fig.add_subplot(333)
ax7.plot(data['train']['regular']['dnn_score']['steps'][:74], [v * 100 for v in data['train']['regular']['dnn_score']['values'][0:74]], 'k')
ax7.set_ylim(bottom=0, top=104)
ax7.set_title('CIFAR-100')
ax7.yaxis.grid()
ax7.set_xticks([0, 5000, 10000, 15000])

ax8 = fig.add_subplot(336)
# ax8.plot(3200, 29.85, 'ro')
ax8.plot(data['test']['regular']['dnn_score']['steps'][:74], [v * 100 for v in data['test']['regular']['dnn_score']['values'][0:74]], 'k')
ax8.set_ylim(bottom=10, top=33)
ax8.yaxis.grid()
ax8.set_xticks([0, 5000, 10000, 15000])

ax9 = fig.add_subplot(339)
ax9.plot(data['train']['regular']['knn_kl_div2_avg']['steps'][:74], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values'][:74]], 'k')
ax9.plot(data['test']['regular']['knn_kl_div2_avg']['steps'][:74] , [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values'][:74]] , 'k--')
# ax9.set_ylim(top=30)
ax9.set_xticks([0, 5000, 10000, 15000])
ax9.yaxis.grid()
ax9.legend(['train', 'test'], loc=(0.77, 0.13))

plt.tight_layout()
plt.savefig('kl_div_overfitting_with_train_acc.png')