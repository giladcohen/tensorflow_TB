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
    logdirs   = (
        '/data/gilad/logs/ma_scores/large_batch/fc2net/mnist/log_1053_080518_ma_score_w_trainset_fc2net_mnist_large_batch_no_aug_wd_0.0_lr_10_schedule_200_2_300_0.4_400_0.08-SUPERSEED=08051800',
        '/data/gilad/logs/ma_scores/large_batch/fc2net/mnist/log_1053_080518_ma_score_w_trainset_fc2net_mnist_large_batch_wd_0.0_lr_10_schedule_200_2_300_0.4_400_0.08-SUPERSEED=08051800'
    )
    run_names = ('test',)
    tag_names = ('knn_score', 'knn_score_trainset', 'ma_score', 'ma_score_trainset', 'md_score', 'md_score_trainset', 'score', 'score_trainset')

    for logdir in logdirs:
        output_dir = os.path.join(logdir, 'data_for_figures')
        mkdir_p(output_dir)
        print("Loading data for logdir: {}".format(logdir))
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
