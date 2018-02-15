from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.misc import collect_features
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import operator

def min_mul_dnn_max_knn_same(agent):
    """
    :param agent: An active learning trainer
    :return: list of indices
    """

    unpool_indices = agent.dataset.get_all_unpool_train_indices()

    pool_features_vec, pool_labels = \
        collect_features(agent=agent,
                         dataset_name='train_pool_eval',
                         fetches=[agent.model.net['embedding_layer'], agent.model.labels],
                         feed_dict={agent.model.dropout_keep_prob: 1.0})

    unpool_features_vec, unpool_predictions_vec = \
        collect_features(agent=agent,
                         dataset_name='train_unpool_eval',
                         fetches=[agent.model.net['embedding_layer'], agent.model.predictions_prob],
                         feed_dict={agent.model.dropout_keep_prob: 1.0})

    agent.log.info('building kNN space only for the labeled (pooled) train features')
    nbrs = KNeighborsClassifier(n_neighbors=30, weights='uniform', p=1)
    nbrs.fit(pool_features_vec, pool_labels)

    agent.log.info('Calculating the estimated labels probability based on KNN')
    estimated_labels_vec = nbrs.predict_proba(unpool_features_vec)
    u_vec = mul_dnn_max_knn_same(agent, estimated_labels_vec, unpool_predictions_vec)

    best_unpool_indices = np.take(unpool_indices, u_vec.argsort()[-agent.dataset.clusters:])
    best_unpool_indices.tolist()
    best_unpool_indices.sort()
    return best_unpool_indices

def most_uncertained_following_min_corr(agent):
    """
    :param agent: An active learning trainer
    :return: list of indices
    """

    unpool_indices = agent.dataset.get_all_unpool_train_indices()

    pool_features_vec, pool_labels = \
        collect_features(agent=agent,
                         dataset_name='train_pool_eval',
                         fetches=[agent.model.net['embedding_layer'], agent.model.labels],
                         feed_dict={agent.model.dropout_keep_prob: 1.0})

    unpool_features_vec, unpool_predictions_vec = \
        collect_features(agent=agent,
                         dataset_name='train_unpool_eval',
                         fetches=[agent.model.net['embedding_layer'], agent.model.predictions_prob],
                         feed_dict={agent.model.dropout_keep_prob: 1.0})

    agent.log.info('building kNN space only for the labeled (pooled) train features')
    nbrs = KNeighborsClassifier(n_neighbors=30, weights='uniform', p=1)
    nbrs.fit(pool_features_vec, pool_labels)

    agent.log.info('Calculating the estimated labels probability based on KNN')
    estimated_labels_vec = nbrs.predict_proba(unpool_features_vec)

    agent.log.info('Calculating the most uncertainty scores based on the DNN output')
    mu_vec = uncertainty_score(agent, unpool_predictions_vec)
    mu_unpool_relative_indices = mu_vec.argsort()[-agent.dataset.clusters*5:]
    mu_unpool_indices = np.take(unpool_indices, mu_unpool_relative_indices)
    mu_unpool_predictions_vec = unpool_predictions_vec[mu_unpool_relative_indices]
    mu_estimated_labels_vec = estimated_labels_vec[mu_unpool_relative_indices]

    agent.log.info('Selecting 1k samples out of the 5k candidate samples')
    corr_vec = correlation_score(agent, mu_estimated_labels_vec, mu_unpool_predictions_vec)
    corr_unpool_relative_indices = corr_vec.argsort()[-agent.dataset.clusters:]
    best_unpool_indices = np.take(mu_unpool_indices, corr_unpool_relative_indices)
    best_unpool_indices.tolist()
    best_unpool_indices.sort()
    return best_unpool_indices

def min_corr_following_most_uncertained(agent):
    """
    :param agent: An active learning trainer
    :return: list of indices
    """

    unpool_indices = agent.dataset.get_all_unpool_train_indices()

    pool_features_vec, pool_labels = \
        collect_features(agent=agent,
                         dataset_name='train_pool_eval',
                         fetches=[agent.model.net['embedding_layer'], agent.model.labels],
                         feed_dict={agent.model.dropout_keep_prob: 1.0})

    unpool_features_vec, unpool_predictions_vec = \
        collect_features(agent=agent,
                         dataset_name='train_unpool_eval',
                         fetches=[agent.model.net['embedding_layer'], agent.model.predictions_prob],
                         feed_dict={agent.model.dropout_keep_prob: 1.0})

    agent.log.info('building kNN space only for the labeled (pooled) train features')
    nbrs = KNeighborsClassifier(n_neighbors=30, weights='uniform', p=1)
    nbrs.fit(pool_features_vec, pool_labels)

    agent.log.info('Calculating the estimated labels probability based on KNN')
    estimated_labels_vec = nbrs.predict_proba(unpool_features_vec)

    agent.log.info('Calculating the correlation scores between the KNN and DNN predictions')
    corr_vec = correlation_score(agent, estimated_labels_vec, unpool_predictions_vec)
    corr_unpool_relative_indices = corr_vec.argsort()[-agent.dataset.clusters*5:]
    corr_unpool_indices = np.take(unpool_indices, corr_unpool_relative_indices)
    corr_unpool_predictions_vec = unpool_predictions_vec[corr_unpool_relative_indices]

    agent.log.info('Selecting 1k samples out of the 5k candidate samples')
    mu_vec = uncertainty_score(agent, corr_unpool_predictions_vec)
    mu_unpool_relative_indices = mu_vec.argsort()[-agent.dataset.clusters:]
    best_unpool_indices = np.take(corr_unpool_indices, mu_unpool_relative_indices)
    best_unpool_indices.tolist()
    best_unpool_indices.sort()
    return best_unpool_indices

def mul_dnn_max_knn_same(agent, y_pred_knn, y_pred_dnn):
    """
    Calculates the uncertainty score based on the multiplication of the highest DNN probability with the
    corresponding KNN probability value.
    :param agent: the trainer/tester
    :param y_pred_knn: np.float32 array of the KNN probability
    :param y_pred_dnn: np.float32 array of all the predictions of the network
    :return: uncertainty score for every vector
    """
    if y_pred_knn.shape != y_pred_dnn.shape:
        err_str = 'y_pred_knn.shape != y_pred_dnn.shape ({}!={})'.format(y_pred_knn.shape, y_pred_dnn.shape)
        agent.log.error(err_str)
        raise AssertionError(err_str)

    score = np.empty(shape=y_pred_knn.shape[0], dtype=np.float32)
    for row in xrange(y_pred_knn.shape[0]):
        dnn_max_ind, dnn_max_val = max(enumerate(y_pred_dnn[row]), key=operator.itemgetter(1))
        knn_max_val              = y_pred_knn[row][dnn_max_ind]
        score[row] = 1 - dnn_max_val * knn_max_val
    return score

def uncertainty_score(agent, y_pred_dnn):
    """
    Calculates the uncertainty score for y_pred_dnn
    :param agent: trainer
    :param y_pred_dnn: np.float32 array of the DNN probability
    :return: uncertainty score for every vector
    """
    return 1 - y_pred_dnn.max(axis=1)

def correlation_score(agent, y_pred_knn, y_pred_dnn):
    """
    Calculates the correlation of the knn and dnn predictions
    :param agent: trainer
    :param y_pred_knn: np.float32 array of the KNN probability
    :param y_pred_dnn: np.float32 array of the DNN probability
    :return: correlation for every two prediction vectors
    """
    if y_pred_knn.shape != y_pred_dnn.shape:
        err_str = 'y_pred_knn.shape != y_pred_dnn.shape ({}!={})'.format(y_pred_knn.shape, y_pred_dnn.shape)
        agent.log.error(err_str)
        raise AssertionError(err_str)

    score = np.empty(shape=y_pred_knn.shape[0], dtype=np.float32)
    for row in xrange(y_pred_knn.shape[0]):
        score[row] = -np.correlate(y_pred_knn[row], y_pred_dnn[row])
    return score
