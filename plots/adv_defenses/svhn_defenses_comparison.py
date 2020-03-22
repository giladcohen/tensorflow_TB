import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

datasets = ['cifar10', 'cifar100', 'svhn']
attacks  = ['fgsm', 'jsma', 'deepfool', 'cw', 'pgd', 'ead']
defenses = ['DkNN', 'LID', 'Mahalanobis', 'NNIF']

dataset = datasets[2]
n_groups = len(attacks)

defense_auc_dict = {dataset: {
    'fgsm'    : {'DkNN': 0.852403897, 'LID': 0.883801565, 'Mahalanobis': 0.98136107 , 'NNIF': 0.910604255},
    'jsma'    : {'DkNN': 0.946141247, 'LID': 0.94312123 , 'Mahalanobis': 0.991461796, 'NNIF': 0.9829059},
    'deepfool': {'DkNN': 0.911326067, 'LID': 0.920042024, 'Mahalanobis': 0.960727957, 'NNIF': 0.971065459},
    'cw'      : {'DkNN': 0.951495776, 'LID': 0.95637593 , 'Mahalanobis': 0.982578844, 'NNIF': 0.986808722},
    'pgd'     : {'DkNN': 0.790695184, 'LID': 0.809220951, 'Mahalanobis': 0.904078486, 'NNIF': 0.92458346},
    'ead'     : {'DkNN': 0.847659606, 'LID': 0.867363151, 'Mahalanobis': 0.929501911, 'NNIF': 0.937201323},
}}

defense_auc_all_layer_dict = {dataset: {
    'fgsm'    : {'LID': 0.999199487, 'Mahalanobis': 0.999978194, 'NNIF': 0.99995706},
    'jsma'    : {'LID': 0.970616271, 'Mahalanobis': 0.999110281, 'NNIF': 0.997622604},
    'deepfool': {'LID': 0.9390103  , 'Mahalanobis': 0.979218036, 'NNIF': 0.99058469},
    'cw'      : {'LID': 0.958228758, 'Mahalanobis': 0.991787393, 'NNIF': 0.995891897},
    'pgd'     : {'LID': 0.801165095, 'Mahalanobis': 0.94469114 , 'NNIF': 0.961761959},
    'ead'     : {'LID': 0.87860518 , 'Mahalanobis': 0.957703782, 'NNIF': 0.973957302},
}}

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
values1 = [defense_auc_dict[dataset]['fgsm']['DkNN'], defense_auc_dict[dataset]['jsma']['DkNN'],
           defense_auc_dict[dataset]['deepfool']['DkNN'], defense_auc_dict[dataset]['cw']['DkNN'],
           defense_auc_dict[dataset]['pgd']['DkNN'], defense_auc_dict[dataset]['ead']['DkNN']]
rects1 = plt.bar(index + bar_width, values1, bar_width,
                 alpha=opacity,
                 color='black',
                 edgecolor='black',
                 label='DkNN')

# defense rectangle: all of LID
values2 = [defense_auc_dict[dataset]['fgsm']['LID'], defense_auc_dict[dataset]['jsma']['LID'],
           defense_auc_dict[dataset]['deepfool']['LID'], defense_auc_dict[dataset]['cw']['LID'],
           defense_auc_dict[dataset]['pgd']['LID'], defense_auc_dict[dataset]['ead']['LID']]
rects2 = plt.bar(index + 2*bar_width, values2, bar_width,
                 alpha=opacity,
                 color='blue',
                 edgecolor='black',
                 label='LID')

# defense rectangle: LID boost for all layers
values22 = [defense_auc_boost_dict[dataset]['fgsm']['LID'], defense_auc_boost_dict[dataset]['jsma']['LID'],
            defense_auc_boost_dict[dataset]['deepfool']['LID'], defense_auc_boost_dict[dataset]['cw']['LID'],
            defense_auc_boost_dict[dataset]['pgd']['LID'], defense_auc_boost_dict[dataset]['ead']['LID']]
rects22 = plt.bar(index + 2*bar_width, values22, bar_width,
                 alpha=opacity,
                 color='blue',
                 edgecolor='black',
                 hatch='//',
                 label='LID',
                 bottom=values2)

# defense rectangle: all of Mahalanobis
values3 = [defense_auc_dict[dataset]['fgsm']['Mahalanobis'], defense_auc_dict[dataset]['jsma']['Mahalanobis'],
           defense_auc_dict[dataset]['deepfool']['Mahalanobis'], defense_auc_dict[dataset]['cw']['Mahalanobis'],
           defense_auc_dict[dataset]['pgd']['Mahalanobis'], defense_auc_dict[dataset]['ead']['Mahalanobis']]
rects3 = plt.bar(index + 3*bar_width, values3, bar_width,
                 alpha=opacity,
                 color='green',
                 edgecolor='black',
                 label='Mahalanobis')

# defense rectangle: Mahalanobis boost
values33 = [defense_auc_boost_dict[dataset]['fgsm']['Mahalanobis'], defense_auc_boost_dict[dataset]['jsma']['Mahalanobis'],
            defense_auc_boost_dict[dataset]['deepfool']['Mahalanobis'], defense_auc_boost_dict[dataset]['cw']['Mahalanobis'],
            defense_auc_boost_dict[dataset]['pgd']['Mahalanobis'], defense_auc_boost_dict[dataset]['ead']['Mahalanobis']]
rects33 = plt.bar(index + 3*bar_width, values33, bar_width,
                 alpha=opacity,
                 color='green',
                 edgecolor='black',
                 hatch='//',
                 label='Mahalanobis',
                 bottom=values3)

# defense rectangle: all of NNIF
values4 = [defense_auc_dict[dataset]['fgsm']['NNIF'], defense_auc_dict[dataset]['jsma']['NNIF'],
           defense_auc_dict[dataset]['deepfool']['NNIF'], defense_auc_dict[dataset]['cw']['NNIF'],
           defense_auc_dict[dataset]['pgd']['NNIF'], defense_auc_dict[dataset]['ead']['NNIF']]
rects4 = plt.bar(index + 4*bar_width, values4, bar_width,
                 alpha=opacity,
                 color='red',
                 edgecolor='black',
                 label='NNIF (ours)')

# defense rectangle: NNIF boost
values44 = [defense_auc_boost_dict[dataset]['fgsm']['NNIF'], defense_auc_boost_dict[dataset]['jsma']['NNIF'],
            defense_auc_boost_dict[dataset]['deepfool']['NNIF'], defense_auc_boost_dict[dataset]['cw']['NNIF'],
            defense_auc_boost_dict[dataset]['pgd']['NNIF'], defense_auc_boost_dict[dataset]['ead']['NNIF']]
rects44 = plt.bar(index + 4*bar_width, values44, bar_width,
                 alpha=opacity,
                 color='red',
                 edgecolor='black',
                 hatch='//',
                 label='Mahalanobis',
                 bottom=values4)

colorless_patch = mpatches.Patch(label='all layers', hatch='//', edgecolor='black', facecolor='white')

plt.xlabel('Attack methods')
plt.ylabel('AUC score')
plt.ylim(bottom=0.76, top=1.04)
plt.xticks(index + 2.5*bar_width, ('FGSM', 'JSMA', 'DeepFool', 'CW', 'PGD', 'EAD'))
plt.yticks([0.76, 0.8, 0.84, 0.88, 0.92, 0.96, 1.0],
           ['0.76', '0.8', '0.84', '0.88', '0.92', '0.96', '1.0'])
plt.title('SVHN')
plt.legend((rects1, rects2, rects3, rects4, colorless_patch), ('DkNN', 'LID', 'Mahanalobis', 'NNIF (ours)', 'all layers'),
           loc=(0.04, 0.88), ncol=3, fancybox=True, prop={'size': 10})
plt.tight_layout()
plt.savefig('svhn_defenses.png', dpi=350)