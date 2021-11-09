import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
# import faiss

from data_classes import SubspaceDataset
from dist_perm import DistPerm

def main():

    n = 1000
    d = 8
    D = 128
    k = 16

    data_source = SubspaceDataset(d, D)
    index_dp = DistPerm(k)
    # index_l2 = faiss.IndexFlatL2(D)
    index_l2 = NearestNeighbors(n_neighbors=k)

    db = data_source.generate_db(n)
    index_dp.fit(db)
    index_dp.add(db)
    data = np.array(db)
    index_l2.fit(data)
    # index_l2.add(np.array(db))

    num_queries = 100
    R = 20
    queries = data_source.generate_queries(num_queries)[0]
    found = index_dp.search(queries, R).numpy()
    # true = index_l2.search(queries, k)
    true = index_l2.kneighbors(queries, n_neighbors=R)[1]

    # precision_R = num_queries*[0]
    avg_precision = num_queries*[0]
    for i in range(num_queries):
        hits_vec = np.array([1*(f in true[i]) for f in found[i]])
        precision_R = np.cumsum(hits_vec)/np.arange(1, R+1)

        avg_precision[i] = sum(hits_vec*precision_R)/R

    print('AP@%d: %.3f +/- %.4f' % (R, np.mean(avg_precision), np.std(avg_precision)))
    print('\nDONE\n')

if __name__ == '__main__':
    main()
