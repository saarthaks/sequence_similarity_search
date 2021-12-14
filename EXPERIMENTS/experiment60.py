import numpy as np
import matplotlib.pyplot as plt
import torch
import faiss

from data_classes import BERTDataset
from dist_perm import DistPerm
import utils

n = 300000
D = 128
int_dim = 6

num_queries = 100
R = 50

ks = [4, 8, 16, 32, 64, 128, 256, 512]
m = 1024

file_q = './datasets/Q.pt'
file_db = './datasets/D.pt'
src = BERTDataset(file_q, file_db, n)
db = src.generate_db()
data = np.array(db).astype(np.float32)
queries = src.generate_queries(num_queries)
quers = np.array(queries).astype(np.float32)

index_l2 = faiss.IndexFlatL2(D)
index_l2.add(data)
true = index_l2.search(quers, R)[1]

index_dp = DistPerm(m, k=ks[-1])
index_dp.fit(db, 'kmeans')

recall_pq = np.zeros((len(ks),))
recall_pq_theor = np.zeros_like(recall_pq)
recall_dp = np.zeros_like(recall_pq)
recall_dp_theor = np.zeros_like(recall_pq)
for i, k in enumerate(ks):
    index_pq = faiss.IndexPQ(D, k // 4, 4)
    index_pq.train(data)
    index_pq.add(data)
    found_pq = index_pq.search(quers, R)[1]
    recall_pq[i] = utils.mean_recall(found_pq, true)[0]
    ratio = R/(n/(2**k))
    recall_pq_theor[i] = ratio / (np.floor(ratio) + 1) 

    index_dp.k = k
    index_dp.add(db)
    found_dp = index_dp.search(queries, R).numpy()
    parts = (np.e*m*(m-1)/2/int_dim)**int_dim
    ratio = R/(n/parts)
    recall_dp[i] = utils.mean_recall(found_dp, true)[0]
    recall_dp_theor[i] = ratio / (np.floor(ratio) + 1)

    print('k = %d' % k, 'PQ Recall = %.3f' % recall_pq[i], 'Theor = %.3f' % recall_pq_theor[i])
    print('k = %d' % k, 'DP Recall = %.3f' % recall_dp[i], 'Theor = %.3f' % recall_dp_theor[i])
    index_dp.clear()

print(recall_pq)
print(recall_dp)
plt.figure()
plt.semilogx(ks, recall_pq, '-r', label='PQ')
plt.semilogx(ks, recall_pq_theor, '--r', label='PQ - Theor.')
plt.semilogx(ks, recall_dp, '-b', label='DP')
plt.semilogx(ks, recall_dp_theor, '--b', label='DP - Theor.')
plt.legend()
plt.savefig('./figures/experiment60.png', bbox_inches='tight')

np.savez('./experiment60_data.npz', pq=recall_pq, pqt = recall_pq_theor, dp=recall_dp, dpt=recall_dp_theor, ks=ks)
