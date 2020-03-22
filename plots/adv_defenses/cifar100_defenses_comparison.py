import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

datasets = ['cifar10', 'cifar100', 'svhn']
attacks  = ['fgsm', 'jsma', 'deepfool', 'cw', 'pgd', 'ead']
defenses = ['DkNN', 'LID', 'Mahalanobis', 'NNIF']

dataset = datasets[1]
n_groups = len(attacks)

defense_auc_dict = {dataset: {
    'fgsm'    : {'DkNN': 0.936460617, 'LID': 0.80680724,  'Mahalanobis': 0.839000834, 'NNIF': 0.872308705},
    'jsma'    : {'DkNN': 0.834596403, 'LID': 0.743310138, 'Mahalanobis': 0.902017972, 'NNIF': 0.86627103},
    'deepfool': {'DkNN': 0.767129816, 'LID': 0.522524931, 'Mahalanobis': 0.620539083, 'NNIF': 0.842032781},
    'cw'      : {'DkNN': 0.937669472, 'LID': 0.678359894, 'Mahalanobis': 0.716018721, 'NNIF': 0.945810123},
    'pgd'     : {'DkNN': 0.737799477, 'LID': 0.722524732, 'Mahalanobis': 0.723558827, 'NNIF': 0.830938255},
    'ead'     : {'DkNN': 0.784222267, 'LID': 0.521038529, 'Mahalanobis': 0.616512185, 'NNIF': 0.724246951},
}}

defense_auc_all_layer_dict = {dataset: {
    'fgsm'    : {'LID': 0.923338851, 'Mahalanobis': 0.998690835, 'NNIF': 0.999591273},
    'jsma'    : {'LID': 0.786300755, 'Mahalanobis': 0.964449688, 'NNIF': 0.974962771},
    'deepfool': {'LID': 0.516112859, 'Mahalanobis': 0.618058845, 'NNIF': 0.77168281},
    'cw'      : {'LID': 0.678270021, 'Mahalanobis': 0.744282082, 'NNIF': 0.965098732},
    'pgd'     : {'LID': 0.737086411, 'Mahalanobis': 0.785337755, 'NNIF': 0.966018618},
    'ead'     : {'LID': 0.511112838, 'Mahalanobis': 0.629258462, 'NNIF': 0.748621486},
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
plt.ylim(bottom=0.5, top=1.075)
plt.xticks(index + 2.5*bar_width, ('FGSM', 'JSMA', 'DeepFool', 'CW', 'PGD', 'EAD'))
plt.title('CIFAR-100')
plt.legend((rects1, rects2, rects3, rects4, colorless_patch), ('DkNN', 'LID', 'Mahanalobis', 'NNIF (ours)', 'all layers'),
           loc=(0.05, 0.88), ncol=3, fancybox=True, prop={'size': 10})
plt.tight_layout()
plt.savefig('cifar100_defenses.png', dpi=350)