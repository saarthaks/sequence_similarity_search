import numpy as np
import matplotlib.pyplot as plt
import torch
import faiss

from data_classes import BERTDataset
from dist_perm import DistPerm
from fly import fly
import utils

def main():

    ks = [4, 8, 16, 32, 64]
    ms = [10*i for i in ks]
    
    ns = [1000, 10000, 100000, 685571]
    D = 128
    R = 100
    num_queries = 50

    fq = './datasets/Q.pt'
    fdb = './datasets/D.pt'

    dp_map = []
    pq_map = []
    fly_map = []
    for n in ns:
        source = BERTDataset(fq, fdb, n)
        db = source.generate_db()
        data = np.array(db).astype(np.float32)
        queries = source.generate_queries(num_queries)
        quers = np.array(queries).astype(np.float32)

        index_l2 = faiss.IndexFlatL2(D)
        index_l2.add(data)
        true = index_l2.search(quers, R)[1]

        MAPs_dp = []
        MAPs_pq = []
        MAPs_fly = []
        for m, k in zip(ms, ks):
            index_dp = DistPerm(m, k=k)
            index_dp.fit(db)
            index_dp.add(db)
            found_dp = index_dp.search(queries, R).numpy()
            MAP_dp = utils.mean_avg_precision(found_dp, true)[0]
            MAPs_dp.append(MAP_dp)

            index_pq = faiss.IndexPQ(D, k // 4, 4)
            index_pq.train(data)
            index_pq.add(data)
            found_pq = index_pq.search(quers, R)[1]
            MAP_pq = utils.mean_avg_precision(found_pq, true)[0]
            MAPs_pq.append(MAP_pq)

            index_fly = fly(k, m)
            index_fly.fit(data, sampling_ratio=0.1)
            index_fly.add(data)
            found_fly = index_fly.search(quers, R)
            MAP_fly = utils.mean_avg_precision(found_fly, true)[0]
            MAPs_fly.append(MAP_fly)

        dp_map.append(MAPs_dp)
        pq_map.append(MAPs_pq)
        fly_map.append(MAPs_fly)

    dp_map = np.array(dp_map)
    pq_map = np.array(pq_map)
    fly_map = np.array(fly_map)

    plt.figure(figsize=(8,6))
    for n, dp, pq, fl in zip(ns, dp_map, pq_map, fly_map):
        plt.figure(figsize=(8,6))
        plt.plot(ks, pq, '-*k', label='PQ, |D| = %d' % n)
        plt.plot(ks, fl, '-vb', label='Fly, |D| = %d' % n)
        plt.plot(ks, dp, '-^r', label='DP, |D| = %d' % n)

        plt.legend()
        plt.xlabel('Num. of Signals')
        plt.ylabel('Mean Average Precision')
        plt.title('Top-100 Retrieval, Averaged over 50 Random Queries')
        plt.savefig('./figures/experiment45_%d.png' % n, bbox_inches='tight')

if __name__ == '__main__':
    main()


