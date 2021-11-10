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
    k = 64

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

    num_anchs = (k*np.arange(1, k+1, 2)).astype(np.int)
    index_dp = DistPerm(max(num_anchs), k=k)
    all_anchs = index_dp.fit(db)

    MAPs = np.zeros_like(num_anchs, dtype=np.float32)
    for i, num in enumerate(num_anchs):
        index_dp.anchors = all_anchs[:num]
        index_dp.add(db)
        found_dp = index_dp.search(queries, R).numpy()
        MAPs[i] = utils.mean_avg_precision(found_dp, true)[0]

        print('Num anchs: %d' % num, 'MAP: %.3f' % MAPs[i])

    plt.figure(figsize=(8,5))
    plt.plot(num_anchs/k, MAPs, '--*', label='DP')
    plt.xlabel('Total number of anchors [k]')
    plt.ylabel('Mean average precision')
    plt.title('BERT Dataset, N = 10000, k = 64')
    plt.legend()
    plt.savefig('./figures/experiment33.png', bbox_inches='tight')
    print('\nDone\n')

if __name__ == '__main__':
    main()
