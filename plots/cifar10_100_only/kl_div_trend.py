"""Plotting the 9 KNN accuracy plots"""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
subpos = np.array([0.35, 0.25, 0.5, 0.4])
fig = plt.figure(figsize=(10.0, 10.0))

# wrn, cifar10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax4 = fig.add_subplot(321)
ax4.plot(data['train']['regular']['dnn_score']['steps'], [v * 100 for v in data['train']['regular']['dnn_score']['values']], 'k')
ax4.plot(data['test']['regular']['dnn_score']['steps'] , [v * 100 for v in data['test']['regular']['dnn_score']['values']] , 'k--')
ax4.set_ylim(bottom=0, top=110)
ax4.set_title('CIFAR-10', fontdict={'fontsize': 18})
ax4.yaxis.grid()
ax4.set_ylabel('accuracy (%)', labelpad=5, fontdict={'fontsize': 18})
ax4.legend(['train', 'test'], loc=(0.6, 0.5), prop={'size': 18})
ax4.tick_params(labelsize=14)
ax4.get_xaxis().set_visible(False)

ax5 = fig.add_subplot(323)
ax5.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'r')
ax5.plot(data['train']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values']], 'b')
ax5.plot(data['train']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax5.yaxis.grid()
ax5.set_yticks([0, 20, 40, 60, 80])
ax5.set_ylabel('accuracy (%)', labelpad=15, fontdict={'fontsize': 18})
ax5.legend(['$k$-NN || DNN', 'SVM || DNN', 'LR || DNN'], prop={'size': 14.5})
ax5.tick_params(labelsize=14)
ax5.get_xaxis().set_visible(False)
subax5 = add_subplot_axes(ax5, subpos + [0.1, 0, 0, 0])
subax5.set_ylim([3, 3.7])
subax5.set_yticks([3, 3.7])
subax5.plot(data['train']['regular']['knn_kl_div2_avg']['steps'][-11:], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values'][-11:]], 'r')
subax5.plot(data['train']['regular']['svm_kl_div2_avg']['steps'][-11:], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values'][-11:]], 'b')
subax5.plot(data['train']['regular']['lr_kl_div2_avg']['steps'][-11:] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values'][-11:]] , 'g')
subax5.tick_params(labelsize=12)
ax5.add_patch(patches.Polygon(xy=np.array([[40000, 7], [24000, 21.5], [47800, 48], [50000, 7]]), closed=True, color='silver'))
ax5.add_patch(patches.Rectangle(xy=(40000, 1), width=10000, height=6, facecolor='moccasin'))

ax6 = fig.add_subplot(325)
ax6.plot(data['test']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']], 'r')
ax6.plot(data['test']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['svm_kl_div2_avg']['values']], 'b')
ax6.plot(data['test']['regular']['lr_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax6.yaxis.grid()
ax6.set_ylabel('accuracy (%)', labelpad=15, fontdict={'fontsize': 18})
ax6.set_yticks([0, 20, 40, 60, 80])
ax6.legend(['$k$-NN || DNN', 'SVM || DNN', 'LR || DNN'], prop={'size': 14.5})
ax6.tick_params(labelsize=14)
subax6 = add_subplot_axes(ax6, subpos - [-0.1, 0.21, 0, 0])
subax6.set_ylim([3.5, 5.3])
subax6.set_yticks([3.5, 5.3])
subax6.plot(data['test']['regular']['knn_kl_div2_avg']['steps'][-11:], [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values'][-11:]], 'r')
subax6.plot(data['test']['regular']['svm_kl_div2_avg']['steps'][-11:], [v * 100 for v in data['test']['regular']['svm_kl_div2_avg']['values'][-11:]], 'b')
subax6.plot(data['test']['regular']['lr_kl_div2_avg']['steps'][-11:] , [v * 100 for v in data['test']['regular']['lr_kl_div2_avg']['values'][-11:]] , 'g')
subax6.tick_params(labelsize=12)
ax6.add_patch(patches.Polygon(xy=np.array([[40000, 7], [23700, 21.5], [47800, 47], [50000, 7]]), closed=True, color='silver'))
ax6.add_patch(patches.Rectangle(xy=(40000, 1), width=10000, height=6, facecolor='moccasin'))

# wrn, cifar100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax7 = fig.add_subplot(322)
ax7.plot(data['train']['regular']['dnn_score']['steps'], [v * 100 for v in data['train']['regular']['dnn_score']['values']], 'k')
ax7.plot(data['test']['regular']['dnn_score']['steps'] , [v * 100 for v in data['test']['regular']['dnn_score']['values']] , 'k--')
ax7.set_ylim(bottom=0, top=110)
ax7.set_title('CIFAR-100', fontdict={'fontsize': 18})
ax7.yaxis.grid()
ax7.legend(['train', 'test'], loc=(0.6, 0.3), prop={'size': 18})
ax7.tick_params(labelsize=14)
ax7.get_xaxis().set_visible(False)

ax8 = fig.add_subplot(324)
ax8.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'r')
ax8.plot(data['train']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values']], 'b')
ax8.plot(data['train']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax8.yaxis.grid()
ax8.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
ax8.tick_params(labelsize=14)
ax8.legend(['$k$-NN || DNN', 'SVM || DNN', 'LR || DNN'], prop={'size': 14.5})
ax8.get_xaxis().set_visible(False)

ax9 = fig.add_subplot(326)
ax9.plot(data['test']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']], 'r')
ax9.plot(data['test']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['svm_kl_div2_avg']['values']], 'b')
ax9.plot(data['test']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['test']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax9.yaxis.grid()
ax9.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
ax9.set_yticks([50, 100, 150, 200])
ax9.tick_params(labelsize=14)
ax9.legend(['$k$-NN || DNN', 'SVM || DNN', 'LR || DNN'], prop={'size': 14.5})

plt.tight_layout()
plt.savefig('kl_div_trend.png', dpi=350)