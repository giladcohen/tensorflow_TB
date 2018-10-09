"""Plotting the 9 KNN accuracy plots"""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
subpos = np.array([0.35, 0.25, 0.5, 0.4])
fig = plt.figure(figsize=(15.0, 8.0))

# wrn, mnist
root_dir = '/data/gilad/logs/metrics/wrn/mnist/log_0049_270818_metrics_w_confidence-SUPERSEED=27081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax1 = fig.add_subplot(331)
ax1.plot(data['train']['regular']['dnn_score']['steps'], [v * 100 for v in data['train']['regular']['dnn_score']['values']], 'k')
ax1.plot(data['test']['regular']['dnn_score']['steps'] , [v * 100 for v in data['test']['regular']['dnn_score']['values']] , 'k--')
ax1.set_ylim(bottom=0, top=110)
ax1.set_title('MNIST')
ax1.yaxis.grid()
ax1.set_ylabel('accuracy (%)', labelpad=5, fontdict={'fontsize': 12})
ax1.legend(['train', 'test'], loc=(0.8, 0.5))

ax2 = fig.add_subplot(334)
ax2.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'r')
ax2.plot(data['train']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values']], 'b')
ax2.plot(data['train']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax2.set_ylabel('$D_{KL}$($k$-NN || DNN) (train)', labelpad=5, fontdict={'fontsize': 12})
ax2.yaxis.grid()
ax2.legend(['$k$-NN || DNN', 'SVM || DNN', 'LR || DNN'])
subax2 = add_subplot_axes(ax2, subpos)
subax2.set_ylim([10.8, 12.1])
subax2.set_yticks([10.8, 12.1])
subax2.plot(data['train']['regular']['knn_kl_div2_avg']['steps'][-21:], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values'][-21:]], 'r')
subax2.plot(data['train']['regular']['svm_kl_div2_avg']['steps'][-21:], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values'][-21:]], 'b')
subax2.plot(data['train']['regular']['lr_kl_div2_avg']['steps'][-21:] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values'][-21:]] , 'g')
ax2.add_patch(patches.Polygon(xy=np.array([[800, 20], [530, 60], [975, 130], [1000, 20]]), closed=True, color='silver'))
ax2.add_patch(patches.Rectangle(xy=(800, 4), width=200, height=15, facecolor='moccasin'))


ax3 = fig.add_subplot(337)
ax3.plot(data['test']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']], 'r')
ax3.plot(data['test']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['svm_kl_div2_avg']['values']], 'b')
ax3.plot(data['test']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['test']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax3.set_ylabel('KL divergence (test)', labelpad=5, fontdict={'fontsize': 12})
ax3.yaxis.grid()
ax3.set_xticks([0, 200, 400, 600, 800, 1000])
ax3.set_yticks([0, 50, 100, 150, 200])
ax3.set_ylabel('$D_{KL}$($k$-NN || DNN) (test)', labelpad=5, fontdict={'fontsize': 12})
ax3.legend(['$k$-NN || DNN', 'SVM || DNN', 'LR || DNN'])
subax3 = add_subplot_axes(ax3, subpos - [0, 0.21, 0, 0])
subax3.set_ylim([10.8, 12.1])
subax3.set_yticks([10.8, 12.1])
subax3.plot(data['train']['regular']['knn_kl_div2_avg']['steps'][-21:], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values'][-21:]], 'r')
subax3.plot(data['train']['regular']['svm_kl_div2_avg']['steps'][-21:], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values'][-21:]], 'b')
subax3.plot(data['train']['regular']['lr_kl_div2_avg']['steps'][-21:] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values'][-21:]] , 'g')
ax3.add_patch(patches.Polygon(xy=np.array([[800, 20], [530, 60], [975, 130], [1000, 20]]), closed=True, color='silver'))
ax3.add_patch(patches.Rectangle(xy=(800, 4), width=200, height=15, facecolor='moccasin'))


# wrn, cifar10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax4 = fig.add_subplot(332)
ax4.plot(data['train']['regular']['dnn_score']['steps'], [v * 100 for v in data['train']['regular']['dnn_score']['values']], 'k')
ax4.plot(data['test']['regular']['dnn_score']['steps'] , [v * 100 for v in data['test']['regular']['dnn_score']['values']] , 'k--')
ax4.set_ylim(bottom=0, top=110)
ax4.set_title('CIFAR-10')
ax4.yaxis.grid()
ax4.legend(['train', 'test'], loc=(0.8, 0.5))

ax5 = fig.add_subplot(335)
ax5.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'r')
ax5.plot(data['train']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values']], 'b')
ax5.plot(data['train']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax5.yaxis.grid()
ax5.set_yticks([0, 20, 40, 60, 80])
ax5.legend(['$k$-NN || DNN', 'SVM || DNN', 'LR || DNN'])

ax6 = fig.add_subplot(338)
ax6.plot(data['test']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']], 'r')
ax6.plot(data['test']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['svm_kl_div2_avg']['values']], 'b')
ax6.plot(data['test']['regular']['lr_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax6.yaxis.grid()
ax6.legend(['$k$-NN || DNN', 'SVM || DNN', 'LR || DNN'])

# wrn, cifar100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax7 = fig.add_subplot(333)
ax7.plot(data['train']['regular']['dnn_score']['steps'], [v * 100 for v in data['train']['regular']['dnn_score']['values']], 'k')
ax7.plot(data['test']['regular']['dnn_score']['steps'] , [v * 100 for v in data['test']['regular']['dnn_score']['values']] , 'k--')
ax7.set_ylim(bottom=0, top=110)
ax7.set_title('CIFAR-10')
ax7.yaxis.grid()
ax7.legend(['train', 'test'], loc=(0.8, 0.4))

ax8 = fig.add_subplot(336)
ax8.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'r')
ax8.plot(data['train']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values']], 'b')
ax8.plot(data['train']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax8.yaxis.grid()
ax8.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
ax8.legend(['$k$-NN || DNN', 'SVM || DNN', 'LR || DNN'])

ax9 = fig.add_subplot(339)
ax9.plot(data['test']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']], 'r')
ax9.plot(data['test']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['svm_kl_div2_avg']['values']], 'b')
ax9.plot(data['test']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['test']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax9.yaxis.grid()
ax9.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
ax9.set_yticks([50, 100, 150, 200])
ax9.legend(['$k$-NN || DNN', 'SVM || DNN', 'LR || DNN'])

plt.tight_layout()
plt.savefig('kl_div_trend.png')