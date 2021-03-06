"""This script extract the scalar for the multiple knn tester, taking knn accuracy for multiple Ks and L1/L2 norms"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import errno
import os
import re

import tensorflow as tf
from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer  # pylint: disable=line-too-long
import tensorflow_TB.plots as plots
import json

# Control downsampling: how many scalar data do we keep for each run/tag
# combination?
SIZE_GUIDANCE = {'scalars': 1000}

def rm_str(str1, str2='_trainset'):
    """
    Removing str2 from str1
    :param str1: string
    :param str2: string
    :return: str1 without str2
    """
    return str1.replace(str2, '')

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

def remove_duplicates(c1, c2):
    """
    :param c1: list1
    :param c2: list2
    :return: None. deletes elements in c1 and c2 if the same element is already found in c1
    """

    assert len(c1) == len(c2), "length of c1 must equal the length of c2"
    i = len(c1) - 1
    while i >= 0:
        if c1[i] in c1[0:i]:
            del c1[i]
            del c2[i]
        i -= 1

def main():

    all_ks = [1, 3, 4, 5, 6, 7, 8, 9, 10,
              12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
              45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
              110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
              220, 240, 260, 280, 300,
              350, 400, 450, 500,
              600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    all_ks.extend(range(1600, 6001, 100))

    logdirs = [
        '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_7k-SUPERSEED=21011903',
        '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_8k-SUPERSEED=21011901',
        '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_9k-SUPERSEED=21011901',
        '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_10k-SUPERSEED=21011901'
    ]
    max_ks = [3500, 4000, 4500, 5000]

    bayesian_applied = False
    num_classes = 2
    # logdirs = []
    # max_ks  = []
    # for i in range(9, 11):
    #     logdir = '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=21011900'.format(i)
    #     logdirs.append(logdir)
    #     max_ks.append(int(i * 1000 / num_classes))

    run_names = ['test']
    reg_tags_dict = {}  # every sample size has its own element (list)

    for i, logdir in enumerate(logdirs):
        reg_tags_dict[logdir] = ['dnn_confidence_avg', 'dnn_confidence_median', 'dnn_score']

        max_k = max_ks[i]
        for k in all_ks:
            for norm in ['L1', 'L2']:
                if k <= max_k:
                    common_str_list = ['knn/k={}/norm={}/'.format(k, norm)]
                    if bayesian_applied:
                        common_str_list.append('knn/dropout=0.5/k={}/norm={}/'.format(k, norm))
                    for common_str in common_str_list:
                        reg_tags_dict[logdir].append(common_str + 'knn_confidence_avg')
                        reg_tags_dict[logdir].append(common_str + 'knn_confidence_median')
                        reg_tags_dict[logdir].append(common_str + 'knn_kl_div_avg')
                        reg_tags_dict[logdir].append(common_str + 'knn_kl_div_median')
                        reg_tags_dict[logdir].append(common_str + 'knn_kl_div2_avg')
                        reg_tags_dict[logdir].append(common_str + 'knn_kl_div2_median')
                        reg_tags_dict[logdir].append(common_str + 'knn_kl_div3_avg')
                        reg_tags_dict[logdir].append(common_str + 'knn_kl_div3_median')
                        reg_tags_dict[logdir].append(common_str + 'knn_kl_div4_avg')
                        reg_tags_dict[logdir].append(common_str + 'knn_kl_div4_median')
                        reg_tags_dict[logdir].append(common_str + 'knn_psame')
                        reg_tags_dict[logdir].append(common_str + 'knn_score')

    for i, logdir in enumerate(logdirs):
        reg_tags = reg_tags_dict[logdir]
        output_dir = os.path.join(logdir, 'data_for_figures')
        mkdir_p(output_dir)
        print("Loading data for logdir: {}".format(logdir))
        multiplexer = create_multiplexer(logdir)
        for run_name in run_names:
            regular_dir = os.path.join(output_dir, 'regular')
            mkdir_p(regular_dir)
            for tag_name in reg_tags:
                output_filename = '%s___%s' % (munge_filename(run_name), munge_filename(tag_name))
                output_filepath = os.path.join(regular_dir, output_filename)
                print("Exporting (run=%r, tag=%r) to %r..." % (run_name, tag_name, output_filepath))
                export_scalars(multiplexer, run_name, tag_name, output_filepath)
        print("Done extracting scalars. Now processing the JSON file")
        data = {}
        data['train']   = {}
        data['test']    = {}

        # build regular data
        data['train']['regular'] = {}
        data['test']['regular']  = {}
        for reg_tag in reg_tags:
            reg_tag = munge_filename(reg_tag)
            if 'trainset' in reg_tag:
                rec = 'train'
            else:
                rec = 'test'
            csv_file = os.path.join(regular_dir, 'test___' + reg_tag)
            data[rec]['regular'][rm_str(reg_tag)] = {}
            data[rec]['regular'][rm_str(reg_tag)]['steps'], data[rec]['regular'][rm_str(reg_tag)]['values'] = \
                plots.load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=4)
            remove_duplicates(data[rec]['regular'][rm_str(reg_tag)]['steps'], data[rec]['regular'][rm_str(reg_tag)]['values'])

        # export to JSON file
        json_file = os.path.join(output_dir, 'data.json')
        with open(json_file, 'w') as fp:
            json.dump(data, fp)

if __name__ == '__main__':
    main()
