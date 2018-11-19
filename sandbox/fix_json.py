import json
import os

logdir = '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111800'
json_file = os.path.join(logdir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
for key in data['train']['regular'].keys():
    print('fixing {}'.format(key))
    del data['train']['regular'][key]['steps'][101]
    del data['train']['regular'][key]['values'][101]
for key in data['test']['regular'].keys():
    print('fixing {}'.format(key))
    del data['test']['regular'][key]['steps'][101]
    del data['test']['regular'][key]['values'][101]

# dump
with open(json_file, 'w') as fp:
    json.dump(data, fp)

logdir = '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111800'
json_file = os.path.join(logdir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
for key in data['train']['regular'].keys():
    print('fixing {}'.format(key))
    del data['train']['regular'][key]['steps'][101]
    del data['train']['regular'][key]['values'][101]
for key in data['test']['regular'].keys():
    print('fixing {}'.format(key))
    del data['test']['regular'][key]['steps'][101]
    del data['test']['regular'][key]['values'][101]

# dump
with open(json_file, 'w') as fp:
    json.dump(data, fp)

