from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv
import errno
import os
import re

import tensorflow as tf
from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer  # pylint: disable=line-too-long


# Control downsampling: how many scalar data do we keep for each run/tag
# combination?
SIZE_GUIDANCE = {'scalars': 1000}


def extract_scalars(multiplexer, run, tag):
    """Extract tabular data from the scalars at a given run and tag.

    The result is a list of 3-tuples (wall_time, step, value).
    """
    tensor_events = multiplexer.Tensors(run, tag)
    return [(event.wall_time, event.step, tf.make_ndarray(event.tensor_proto).item()) for event in tensor_events]

def create_multiplexer(logdir):
    multiplexer = event_multiplexer.EventMultiplexer(tensor_size_guidance=SIZE_GUIDANCE)
    multiplexer.AddRunsFromDirectory(logdir)
    multiplexer.Reload()
    return multiplexer

def export_scalars(multiplexer, run, tag, filepath, write_headers=True):
    data = extract_scalars(multiplexer, run, tag)
    with open(filepath, 'w') as outfile:
        writer = csv.writer(outfile)
    if write_headers:
        writer.writerow(('wall_time', 'step', 'value'))
    for row in data:
        writer.writerow(row)

NON_ALPHABETIC = re.compile('[^A-Za-z0-9_]')

def munge_filename(name):
    """Remove characters that might not be safe in a filename."""
    return NON_ALPHABETIC.sub('_', name)

def mkdir_p(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if not (e.errno == errno.EEXIST and os.path.isdir(directory)):
            raise

def main():
    run_names = ('test',)
    tag_names = ('knn_score',)

      # '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800',
      # '/data/gilad/logs/ma_scores/wrn/cifar100/log_1444_070318_wrn_cifar100_ma_score_wd_0.00078-SUPERSEED=07031800',
      # '/data/gilad/logs/ma_scores/lenet/cifar10/log_2354_060318_lenet_ma_score_wd_0.008-SUPERSEED=06031800',
      # '/data/gilad/logs/ma_scores/lenet/cifar100/log_2340_090318_lenet_cifar100_wd_0.01-SUPERSEED=08031800',
      # '/data/gilad/logs/ma_scores/lenet/mnist/log_2200_100318_ma_score_lenet_mnist_wd_0.0-SUPERSEED=10031800',
      # '/data/gilad/logs/ma_scores/fc2net/cifar10/log_1705_090318_ma_score_fc2net_cifar10_wd_0.0-SUPERSEED=08031800',
      # '/data/gilad/logs/ma_scores/fc2net/cifar100/log_1353_100318_ma_score_fc2net_cifar100_wd_0.0-SUPERSEED=10031800',
      # '/data/gilad/logs/ma_scores/fc2net/mnist/log_1409_140318_ma_score_fc2net_mnist_wd_0.0-SUPERSEED=14031800'

    logdir = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800'
    output_dir = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures'
    mkdir_p(output_dir)

    print("Loading data...")
    multiplexer = create_multiplexer(logdir)
    for run_name in run_names:
        for tag_name in tag_names:
            output_filename = '%s___%s' % (munge_filename(run_name), munge_filename(tag_name))
            output_filepath = os.path.join(output_dir, output_filename)
            print("Exporting (run=%r, tag=%r) to %r..." % (run_name, tag_name, output_filepath))
            export_scalars(multiplexer, run_name, tag_name, output_filepath)
    print("Done.")

if __name__ == '__main__':
    main()
