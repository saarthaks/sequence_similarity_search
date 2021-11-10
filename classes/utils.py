import numpy as np
from sklearn.neighbors import NearestNeighbors

def id_estimator(k_value, dataset_size, distances):
    estimates = []
    for i in range(dataset_size):
        k_distances = np.array(distances[i])
        if np.all(k_distances[1:-1]) == False:
            continue

        k_distances_normalized = np.array((k_value - 1)*[k_distances[-1]])/k_distances[1:-1]
        k_distances_normalized_log = np.log(k_distances_normalized)
        k_distance_id_estimate = (1/(k_value - 2)) * np.sum(k_distances_normalized_log, axis=-1)
        estimates.append(k_distance_id_estimate)

    estimates_avg_id = (1/dataset_size) * np.sum( np.array(estimates) )
    id_estimated = 1/estimates_avg_id

    return id_estimated

def id_mle_estimator(dataset, k_list=[2,3,5,10,20]):
    id_estimates = { val:[] for val in k_list }
    dataset_size = dataset.shape[0]

    classifier = NearestNeighbors(n_neighbors=max(k_list)+1, algorithm='ball_tree')
    classifier.fit(dataset)

    distances, indices = classifier.kneighbors(dataset)

    for k in k_list:
        id_estimates[k] = id_estimator(k, dataset_size, distances[:, :k+1])

    return id_estimates

def avg_precision(found, truth):
    R = len(truth)
    hits = np.array([f in truth for f in found])
    precision = np.cumsum(hits)/np.arange(1, R+1)
    avg_pre = np.sum(hits*precision)/R

    return avg_pre

def mean_avg_precision(all_found, all_true):
    num_queries = all_found.shape[0]
    avg_precisions = [avg_precision(f, t) for f, t in zip(all_found, all_true)]

    return np.mean(avg_precisions), np.std(avg_precisions)/np.sqrt(num_queries)
