import numpy as np
import matplotlib.pyplot as plt
import torch
import faiss
import lshfly

from data_classes import BERTDataset
from dist_perm import DistPerm
import utils

def main():
    
    n = 10000
    D = 128
    num_queries = 50
    R = 100
    k = 8
    num_anchs = [8,9,10,11,12]

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

    MAPs_dp = np.zeros_like(num_anchs, dtype=np.float32)
    MAPs_fly = np.zeros_like(num_anchs, dtype=np.float32)
    for i, num in enumerate(num_anchs):
        index_dp = DistPerm(num, k=k)
        index_dp.fit(db)
        index_dp.add(db)
        found_dp = index_dp.search(queries, R).numpy()
        MAPs_dp[i] = utils.mean_avg_precision(found_dp, true)[0]

        index_fly = lshfly.flylsh(data, hash_length=k, sampling_ratio=0.1, embedding_size=num)
        MAPs_fly[i] = index_fly.findmAP(nnn=R, n_points=num_queries)

        print('Num anchs: %d' % num, 'MAP [DP]: %.3f' % MAPs_dp[i], 'MAP [Fly]: %.3f' % MAPs_fly[i])

    plt.figure(figsize=(8,5))
    plt.plot(num_anchs, MAPs_dp, '--*', label='DP')
    plt.plot(num_anchs, MAPs_fly, '--*', label='Fly')
    plt.xlabel('Total number of anchors [k]')
    plt.ylabel('Mean average precision')
    plt.title('BERT Dataset, N = 10000, k = 8')
    plt.legend()
    plt.savefig('./figures/experiment34.png', bbox_inches='tight')
    print('\nDone\n')

if __name__ == '__main__':
    main()
