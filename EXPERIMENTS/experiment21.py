import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
import faiss

from data_classes import BERTDataset
from dist_perm import DistPerm
import utils

def main():

    n = 10000
    D = 128
    k = 256

    num_queries = 100
    R = 200
    ds = [16, 32]

    file_q = './datasets/Q.pt'
    file_db = './datasets/D.pt'
    data_source = BERTDataset(file_q, file_db, n)
    db = data_source.generate_db()
    data = np.array(db).astype(np.float32)
    queries = data_source.generate_queries(num_queries)
    quers = np.array(queries).astype(np.float32)
    index_l2 = faiss.IndexFlatL2(D)
    index_nn = NearestNeighbors(n_neighbors=R)
    index_l2.add(data)
    index_nn.fit(data)
    true_l2 = index_l2.search(quers, R)[1]
    true_nn = index_nn.kneighbors(quers, n_neighbors=R)[1]
    
    for d in ds:
        index_dp = DistPerm(k, d=d)
        index_pq = faiss.IndexPQ(D, d, 4)

        index_dp.fit(db)
        index_dp.add(db)
        index_pq.train(data)
        index_pq.add(data)

        found_dp = index_dp.search(queries, R).numpy()
        found_pq = index_pq.search(quers, R)[1]


        map_dp_l2 = utils.mean_avg_precision(found_dp, true_l2)
        map_dp_nn = utils.mean_avg_precision(found_dp, true_nn)
        map_pq_l2 = utils.mean_avg_precision(found_pq, true_l2)
        map_pq_nn = utils.mean_avg_precision(found_pq, true_nn)

        print(map_dp_l2, map_dp_nn)
        print(map_pq_l2, map_pq_nn)

if __name__ == '__main__':
    main()
