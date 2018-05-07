"""Plotting the 3 P_SAME plots for MNIST, CIFAR-10 and CIFAR-100. Every subplot has 2 plots for the train set and the test set."""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

subpos = [0.4, 0.5, 0.48, 0.3]
fig = plt.figure(figsize=(14, 4))

# wrn, mnist
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___ma_score_trainset'
steps, ma_values_train = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___md_score_trainset'
steps, md_values_train = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___score_trainset'
steps, acc_values_train = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_mnist_wrn_train = [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values_train, md_values_train, acc_values_train)]
P_SAME_mnist_wrn_train = [round(elem, 4) for elem in P_SAME_mnist_wrn_train]


plt.plot(P_SAME_mnist_wrn_train)

