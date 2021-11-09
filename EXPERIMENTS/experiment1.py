import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
import faiss

from data_classes import SubspaceDataset
from dist_perm import DistPerm

def main():

    n = 1024
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
    index_l2.fit(np.array(db))
    # index_l2.add(np.array(db).astype(np.float32))

    num_queries = 20
    R = 20
    queries = data_source.generate_queries(num_queries)[0]
    quers = np.array(queries).astype(np.float32)
    found = index_dp.search(queries, R).numpy()
    # true = index_l2.search(quers, R)[1]
    true = index_l2.kneighbors(queries, n_neighbors=R)[1]


    recall = num_queries*[0]
    for i in range(num_queries):
        recall[i] = sum([1*(f in true[i]) for f in found[i]])/R

    print('Recall@%d: %.3f +/- %.4f' % (R, np.mean(recall), np.std(recall)))

    print('DONE')

if __name__ == '__main__':
    main()
