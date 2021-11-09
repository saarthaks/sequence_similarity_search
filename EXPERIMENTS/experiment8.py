import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
import faiss

from data_classes import SIFTDataset
from dist_perm import DistPerm

def main():

    n = 10000
    D = 128
    k = 128
    d = 120

    num_queries = 100
    R = 200

    data_file = './datasets/SIFT10Mfeatures.mat'
    data_source = SIFTDataset(data_file, n)
    index_dp = DistPerm(k, d=d)

    db = data_source.generate_db()

    index_lsh = faiss.IndexLSH(D, R)
    index_l2 = faiss.IndexFlatL2(D)

    index_dp.fit(db)
    index_dp.add(db)
    data = np.array(db).astype(np.float32)
    index_lsh.train(data)
    index_lsh.add(data)
    index_l2.add(data)

    queries = data_source.generate_queries(num_queries)
    quers = np.array(queries).astype(np.float32)
    found = index_dp.search(queries, R).numpy()
    found2 = index_lsh.search(quers, R)[1]
    true = index_l2.search(quers, R)[1]

    recall = num_queries*[0]
    recall2 = num_queries*[0]
    for i in range(num_queries):
        hits_vec = np.array([1*(f in true[i]) for f in found[i]])
        hits_vec2 = np.array([1*(f in true[i]) for f in found2[i]])

        recall[i] = sum(hits_vec)/R
        recall2[i] = sum(hits_vec2)/R

    print('Recall@%d: %.3f +/- %.4f' % (R, np.mean(recall), np.std(recall)))
    print('Recall@%d: %.3f +/- %.4f\n' % (R, np.mean(recall2), np.std(recall2)))
    print('\nDONE\n')

if __name__ == '__main__':
    main()
