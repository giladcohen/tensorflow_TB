import numpy as np
import matplotlib.pyplot as plt

datasets = ['cifar10', 'cifar100', 'svhn']
attacks  = ['fgsm', 'jsma', 'deepfool', 'cw']
defenses = ['DkNN', 'LID', 'Mahalanobis', 'NNIF']

n_groups = len(attacks)

defense_auc_dict = {'cifar10': {
    'fgsm'    : {'DkNN': 0.878112873, 'LID': 0.901154783, 'Mahalanobis': 0.967997941, 'NNIF': 0.877478261},
    'jsma'    : {'DkNN': 0.953729236, 'LID': 0.946651551, 'Mahalanobis': 0.98948256 , 'NNIF': 0.976712116},
    'deepfool': {'DkNN': 0.958217445, 'LID': 0.954346369, 'Mahalanobis': 0.964877019, 'NNIF': 0.998185341},
    'cw'      : {'DkNN': 0.968843133, 'LID': 0.976555765, 'Mahalanobis': 0.969558577, 'NNIF': 0.990535253}}}

defense_auc_all_layer_dict = {'cifar10': {
    'fgsm'    : {'LID': 0.981785979, 'Mahalanobis': 0.997981407},
    'jsma'    : {'LID': 0.957410585, 'Mahalanobis': 0.995612383},
    'deepfool': {'LID': 0.957988513, 'Mahalanobis': 0.97491799},
    'cw'      : {'LID': 0.978174446, 'Mahalanobis': 0.964752135}}}

defense_auc_boost_dict = {}
for key1 in defense_auc_all_layer_dict.keys():
    defense_auc_boost_dict[key1] = {}
    for key2 in defense_auc_all_layer_dict[key1].keys():
        defense_auc_boost_dict[key1][key2] = {}
        for key3 in defense_auc_all_layer_dict[key1][key2].keys():
            defense_auc_boost_dict[key1][key2][key3] = \
                np.maximum(0.0, defense_auc_all_layer_dict[key1][key2][key3] - defense_auc_dict[key1][key2][key3])

fig, ax = plt.subplots(figsize=(5, 5))

index = np.arange(n_groups)
bar_width = 0.15

opacity = 0.4

# defense rectangle: all of DkNN
values1 = [defense_auc_dict['cifar10']['fgsm']['DkNN'], defense_auc_dict['cifar10']['jsma']['DkNN'],
          defense_auc_dict['cifar10']['deepfool']['DkNN'], defense_auc_dict['cifar10']['cw']['DkNN']]
rects1 = plt.bar(index + bar_width, values1, bar_width,
                 alpha=opacity,
                 color='black',
                 label='DkNN')

# defense rectangle: all of LID
values2 = [defense_auc_dict['cifar10']['fgsm']['LID'], defense_auc_dict['cifar10']['jsma']['LID'],
          defense_auc_dict['cifar10']['deepfool']['LID'], defense_auc_dict['cifar10']['cw']['LID']]
rects2 = plt.bar(index + 2*bar_width, values2, bar_width,
                 alpha=opacity,
                 color='blue',
                 label='LID')

# defense rectangle: LID boost for all layers
values22 = [defense_auc_boost_dict['cifar10']['fgsm']['LID'], defense_auc_boost_dict['cifar10']['jsma']['LID'],
          defense_auc_boost_dict['cifar10']['deepfool']['LID'], defense_auc_boost_dict['cifar10']['cw']['LID']]
rects22 = plt.bar(index + 2*bar_width, values22, bar_width,
                 alpha=opacity,
                 color='pink',
                 label='LID',
                 bottom=values2)

# defense rectangle: all of Mahalanobis
values3 = [defense_auc_dict['cifar10']['fgsm']['Mahalanobis'], defense_auc_dict['cifar10']['jsma']['Mahalanobis'],
          defense_auc_dict['cifar10']['deepfool']['Mahalanobis'], defense_auc_dict['cifar10']['cw']['Mahalanobis']]
rects3 = plt.bar(index + 3*bar_width, values3, bar_width,
                 alpha=opacity,
                 color='green',
                 label='Mahalanobis')

# defense rectangle: Mahalanobis boost
values33 = [defense_auc_boost_dict['cifar10']['fgsm']['Mahalanobis'], defense_auc_boost_dict['cifar10']['jsma']['Mahalanobis'],
          defense_auc_boost_dict['cifar10']['deepfool']['Mahalanobis'], defense_auc_boost_dict['cifar10']['cw']['Mahalanobis']]
rects33 = plt.bar(index + 3*bar_width, values33, bar_width,
                 alpha=opacity,
                 color='pink',
                 label='Mahalanobis',
                 bottom=values3)

# defense rectangle: all of NNIF
values4 = [defense_auc_dict['cifar10']['fgsm']['NNIF'], defense_auc_dict['cifar10']['jsma']['NNIF'],
          defense_auc_dict['cifar10']['deepfool']['NNIF'], defense_auc_dict['cifar10']['cw']['NNIF']]
rects4 = plt.bar(index + 4*bar_width, values4, bar_width,
                 alpha=opacity,
                 color='red',
                 label='NNIF (ours)')

plt.xlabel('Attack methods')
plt.ylabel('AUC score')
plt.ylim(bottom=0.85, top=1.025)
plt.xticks(index + 2.5*bar_width, ('FGSM', 'JSMA', 'DeepFool', 'CW'))
plt.yticks([0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0],
           ['0.86', '0.88', '0.9', '0.92', '0.94', '0.96', '0.98', '1.0'])
plt.title('CIFAR-10')
plt.legend((rects1, rects2, rects3, rects4, rects22), ('DkNN', 'LID', 'Mahanalobis', 'NNIF (ours)', 'all layers'),
           loc=(0.05, 0.88), ncol=3, fancybox=True, prop={'size': 10})
plt.tight_layout()
plt.savefig('cifar10_defenses.png', dpi=350)