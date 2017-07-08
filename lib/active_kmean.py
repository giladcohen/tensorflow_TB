'''This function builds the input pool for the active learning algorithm'''

import tensorflow as tf
from sklearn.cluster import KMeans, k_means_
from sklearn.cluster.k_means_ import *
# from sklearn.datasets import make_blobs #for testing
import numpy as np
import math
# import matplotlib.pyplot as plt #for debug

def center_updater(init, fixed_centers, n_fixed):
    init[0:n_fixed] = fixed_centers
    return init

'''Testing new class'''
class KMeansWrapper(KMeans):
    def __init__(self, fixed_centers, n_clusters=8, init='k-means++', n_init=1,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=1, algorithm='auto'):
        super(KMeansWrapper, self).__init__(n_clusters=n_clusters, init=init, n_init=n_init,
                 max_iter=max_iter, tol=tol, precompute_distances=precompute_distances,
                 verbose=verbose, random_state=random_state, copy_x=copy_x,
                 n_jobs=n_jobs, algorithm=algorithm)
        self.fixed_centers = fixed_centers
        self.n_fixed = fixed_centers.shape[0]
        assert self.n_fixed < self.n_clusters
    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)
        tol_rev = k_means_._tolerance(X, self.tol)
        itr = 0
        init = k_means_._init_centroids(X, self.n_clusters, 'random')
        self.cluster_centers_ = center_updater(init, self.fixed_centers, self.n_fixed)
        self.inertia_      = np.infty
        self.inertia_prev_ = np.infty
        inertia_del        = np.infty
        while (itr < self.max_iter and inertia_del > tol_rev):
            self.inertia_prev_ = self.inertia_
            self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
                k_means(
                    X, n_clusters=self.n_clusters, init=self.cluster_centers_,
                    n_init=self.n_init, max_iter=1, verbose=self.verbose,
                    precompute_distances=self.precompute_distances,
                    tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                    n_jobs=self.n_jobs, algorithm=self.algorithm,
                    return_n_iter=True)
            self.cluster_centers_ = center_updater(self.cluster_centers_, self.fixed_centers, self.n_fixed)
            if (itr > 0):
                inertia_del = math.fabs((self.inertia_ - self.inertia_prev_) / self.inertia_prev_)
            if (self.verbose):
                print('calculating for itr=%0d: inertia_del = %f, tol_rev = %f' % (itr, inertia_del, tol_rev))
            itr += 1
        if (itr < self.max_iter):
            print('convergence achieved for iteration %0d. inertia = %f. inertia_del = %f' % (itr, self.inertia_, inertia_del))
        else:
            print('convergence not achieved. itr = %0d. inertia = %f. inertia_del = %f' %(itr, self.inertia_, inertia_del))
        return self
    def fit_predict_centers(self, X):
        return self.fit(X).cluster_centers_


# testing
# n_samples = 150000
# random_state = 170
# X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=np.array([[-6.0,-6.0], [-6.0, 6.0], [6.0 , -6.0], [6.0, 6.0]]))
# plt.subplot(221)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# #fixed_centers = np.array([[0.0, -6.0], [0.0, 6.0]])
# fixed_centers = np.array([[1e6, 0.0], [-1e6, 0.0]])
# KM = KMeansWrapper(fixed_centers=fixed_centers, n_clusters=4, random_state=random_state, verbose=1)
# centers = KM.fit_predict_centers(X)
# y_pred = KM.predict(X)
# plt.subplot(222)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
