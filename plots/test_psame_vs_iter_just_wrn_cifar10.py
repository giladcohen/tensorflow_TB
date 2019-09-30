"""Plotting the 9 KNN accuracy plots"""
from tensorflow_TB.utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(15.0, 8.0))
subpos = [0.55, 0.3, 0.5, 0.4]

# wrn, cifar10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax2 = fig.add_subplot(111)
ax2.plot(data['test']['regular']['knn_psame']['steps'], data['test']['regular']['knn_psame']['values'], 'r')
ax2.plot(data['test']['regular']['svm_psame']['steps'], data['test']['regular']['svm_psame']['values'], 'b')
ax2.plot(data['test']['regular']['lr_psame']['steps'] , data['test']['regular']['lr_psame']['values'] , 'g')

ax2.set_ylim(bottom=0, top=1.05)
ax2.set_title('CIFAR-10')
ax2.yaxis.grid()
ax2.set_ylabel('$P_{SAME}$', labelpad=0.5, fontdict={'fontsize': 24})
ax2.tick_params(labelsize=20)
ax2.set_xlabel('iteration step', labelpad=0.5, fontdict={'fontsize': 20})
ax2.legend(['DNN<->$k$-NN', 'DNN<->SVM', 'DNN<->Logistic regression'], loc=(0.6, 0.62), fontsize=20)

# fix subax position
# pos11 = subax1.get_position() # [[0.250367647059, 0.721470588235], [0.341544117647, 0.789411764706]]
# pos12 = [pos11.x0-0.052, pos11.y0+0.05, pos11.width, pos11.height]
# subax1.set_position(pos12)
# pos41 = subax4.get_position() # [[0.250367647059, 0.449705882353], [0.364338235294, 0.540294117647]]
# pos42 = [pos41.x0-0.052, pos41.y0, pos41.width, pos41.height]
# subax4.set_position(pos42)
plt.tight_layout()
plt.savefig('test_psame_vs_iter_just_wrn_cifar10.png')

