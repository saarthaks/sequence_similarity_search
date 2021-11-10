import numpy as np
import matplotlib.pyplot as plt
import torch
import faiss
import lshfly

from data_classes import BERTDataset
from dist_perm import DistPerm
import utils

def main():

    n = 100000
    D = 128
    num_queries = 20
    R = 1000
    ks = [8, 16, 32, 64, 128, 256, 512]
    num_anchs = [k*np.arange(1,5) for k in ks]

    file_q = './datasets/Q.pt'
    file_db = './datasets/D.pt'
    source = BERTDataset(file_q, file_db, n)
    db = source.generate_db()
    data = np.array(db).astype(np.float32)
    queries = source.generate_queries(num_queries)
    quers = np.array(queries).astype(np.float32)

    index_l2 = faiss.IndexFlatL2(D)
    index_l2.add(data)
    true = index_l2.search(quers, R)[1]

    MAPs_dp = np.zeros((len(ks), len(num_anchs[0])))
    MAPs_fly = np.zeros_like(MAPs_dp)
    for i, k in enumerate(ks):
        for j, num in enumerate(num_anchs[i]):
            index_dp = DistPerm(num, k=k)
            index_dp.fit(db)
            index_dp.add(db)
            index_fly = lshfly.flylsh(data, hash_length=k, sampling_ratio=0.1, embedding_size=num)

            found_dp = index_dp.search(queries, R).numpy()
            MAPs_dp[i,j] = utils.mean_avg_precision(found_dp, true)[0]
            MAPs_fly[i,j] = index_fly.findmAP(nnn=R, n_points=num_queries)
            print('DP MAP: %.3f' % MAPs_dp[i,j], 'Fly MAP: %.3f' % MAPs_fly[i,j])

    np.savez('./experiment36_res.npz', MAPs_dp=MAPs_dp, MAPs_fly=MAPs_fly)

if __name__ == '__main__':
    main()
