'''This function builds the input pool for the active learning algorithm'''
from __future__ import division

from sklearn.cluster import KMeans, k_means_
from sklearn.cluster.k_means_ import *
import lib.logger.logger as logger
import numpy as np
import math

def center_updater(init, fixed_centers, n_fixed):
    init[0:n_fixed] = fixed_centers
    return init

'''Testing new class'''
class KMeansWrapper(KMeans):
    def __init__(self, name, prm, fixed_centers, *args, **kwargs):
        super(KMeansWrapper, self).__init__(*args, **kwargs)
        self.name = name
        self.prm = prm
        self.log = logger.get_logger(name)

        self.fixed_centers = fixed_centers
        self.n_fixed = fixed_centers.shape[0]
        self.init = 'random'
        self.n_init = 1
        self.verbose = True
        self.assert_config()

    def __str__(self):
        return self.name

    def assert_config(self):
        if self.n_fixed >= self.n_clusters:
            err_str = 'number of fixed centers ({}) must be smaller than n_clusters ({})'.format(self.n_fixed, self.n_clusters)
            self.log.error(err_str)
            raise AssertionError(err_str)

    def fit(self, X, y=None):
        # FIXME(gilad): sub-optimal. consider using _kmeans_single_elkan.
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)
        tol = k_means_._tolerance(X, self.tol)
        itr = 0
        init = k_means_._init_centroids(X, self.n_clusters, 'random', random_state)
        self.cluster_centers_ = center_updater(init, self.fixed_centers, self.n_fixed)
        self.inertia_      = np.infty
        self.inertia_prev_ = np.infty
        inertia_del        = np.infty
        while itr < self.max_iter and inertia_del > tol:
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
            if itr > 0:
                inertia_del = math.fabs((self.inertia_ - self.inertia_prev_) / self.inertia_prev_)
            if self.verbose:
                self.log.info('calculating for itr={}: inertia_del={}, tol={}'.format(itr, inertia_del, tol))
            itr += 1
        if itr < self.max_iter:
            self.log.info('convergence achieved for iteration {}. inertia={}. inertia_del={}'.format(itr, self.inertia_, inertia_del))
        else:
            self.log.info('convergence not achieved. itr={}. inertia={}. inertia_del={}'.format(itr, self.inertia_, inertia_del))
        return self

    def fit_predict_centers(self, X):
        return self.fit(X).cluster_centers_
