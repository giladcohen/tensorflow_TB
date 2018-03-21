from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import csv

def plot_embedding(X, Y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(Y[i]),
                 color=plt.cm.Set1(Y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        ## only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                ## don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def plot_embedding2(vis_x, vis_y, c, title=None):
    plt.figure()
    plt.scatter(vis_x, vis_y, c=c, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    # plt.colorbar(ticks=['airplane', 'automobile', 'bird',
    #                     'cat', 'deer', 'dog', 'frog', 'horse',
    #                     'ship', 'truck'])
    plt.clim(-0.5, 9.5)
    if title is not None:
        plt.title(title)
    plt.show()

def load_data_from_csv(csv_file):
    """
    Importing 'steps' and 'values' from a csv file containing header with: wall_time, step, value
    :param csv_file: path to csv file
    :return: (steps, values) tuple
    """
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f)
        headers = reader.next()
        data = {}
        for h in headers:
            data[h] = []
        for row in reader:
            for h, v in zip(headers, row):
                data[h].append(v)
        assert len(data['step']) == len(data['value'])

    # convert steps to int and values to float
    data['step']  = [int(elem)   for elem in data['step']]
    data['value'] = [float(elem) for elem in data['value']]
    return data['step'], data['value']

def load_data_from_csv_wrapper(csv_file):
    """wrapper to fetch corrected values from csv_file"""
    steps, values = load_data_from_csv(csv_file)
    values = [round(100 * elem, 2) for elem in values]
    return steps, values



