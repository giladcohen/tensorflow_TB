from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.cluster import KMeans, k_means
import lib.logger.logger as logger
import numpy as np


def center_updater(init, fixed_centers, n_fixed):
    init[0:n_fixed] = fixed_centers
    return init

class AllCentersKMeans(KMeans):

    def __init__(self, name, prm, fixed_centers, *args, **kwargs):
        super(AllCentersKMeans, self).__init__(*args, **kwargs)
        self.name = name
        self.prm = prm
        self.log = logger.get_logger(name)

        self.fixed_centers = fixed_centers
        self.n_fixed = fixed_centers.shape[0]
        self.init = 'random'
        self.n_init = 10
        self.verbose = False
        self.assert_config()

    def __str__(self):
        return self.name

    def assert_config(self):
        if self.n_fixed >= self.n_clusters:
            err_str = 'number of fixed centers ({}) must be smaller than n_clusters ({})'.format(self.n_fixed, self.n_clusters)
            self.log.error(err_str)
            raise AssertionError(err_str)

    def fit(self, X, y=None):
        X = self._check_fit_data(X)

        best_centers, best_labels, best_inertia, best_n_iter = None, None, None, None
        for it in range(self.n_init):
            self.log.info('calculating KMeans number #{} out of {}...'.format(it+1, self.n_init))
            centers, labels, _, n_iter = \
                k_means(
                X, n_clusters=self.n_clusters, init=self.init,
                n_init=1, max_iter=self.max_iter, verbose=self.verbose,
                precompute_distances=self.precompute_distances,
                tol=self.tol, random_state=self.random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs, algorithm=self.algorithm,
                return_n_iter=True)
            centers = center_updater(centers, self.fixed_centers, self.n_fixed)
            inertia = np.sum((X - centers[labels]) ** 2, dtype=np.float64)
            self.log.info('Done calculating. Inertia={}, n_iters={}'.format(inertia, n_iter))
            if best_inertia is None or inertia < best_inertia:
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_inertia = inertia
                best_n_iter = n_iter

        self.log.info('best_inertia={}, best_n_iter={}'.format(best_inertia, best_n_iter))
        self.cluster_centers_ = best_centers
        self.labels_          = best_labels
        self.inertia_         = best_inertia
        self.n_iter_          = best_n_iter
        return self
