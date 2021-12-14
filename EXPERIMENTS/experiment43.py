import numpy as np
import matplotlib.pyplot as plt
import torch
import faiss

from data_classes import BERTDataset
from dist_perm import DistPerm
from fly import fly
import utils

def main():
    
    n = 600000
    D = 128
    num_queries = 50
    R = 100
    k = 20
    m = 400

    file_q = './datasets/Q.pt'
    file_db = './datasets/D.pt'
    data_source = BERTDataset(file_q, file_db, n)
    db = data_source.generate_db()
    data = np.array(db).astype(np.float32)
    queries = data_source.generate_queries(num_queries)
    quers = np.array(queries).astype(np.float32)

    index_l2 = faiss.IndexFlatL2(D)
    index_l2.add(data)
    true = index_l2.search(quers, R)[1]

    index_dp = DistPerm(m, k=k)
    index_dp.fit(db)
    index_dp.add(db)
    found_dp = index_dp.search(queries, R).numpy()
    MAP_dp = utils.mean_avg_precision(found_dp, true)[0]

    index_fly = fly(k, m)
    index_fly.fit(data, sampling_ratio=0.1)
    index_fly.add(data)
    found_fly = index_fly.search(quers, R)
    MAP_fly = utils.mean_avg_precision(found_fly, true)[0]

    print('n = %d' % n, 'k = %d' % k, 'm = %d' % m, 'MAP [DP] = %.2f' % MAP_dp, 'MAP [Fly] = %.2f' % MAP_fly)
    print('\nDone\n')

if __name__ == '__main__':
    main()
