import numpy as np
import matplotlib.pyplot as plt
import torch
import faiss

from data_classes import BERTDataset
from dist_perm import DistPerm
import utils

n = 600000
D = 128

num_queries = 100
R = 50

ks = [4, 8, 16, 32, 64, 128, 256, 512]
m = 2048

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

recall_pq = np.zeros((len(ks),))
for i, k in enumerate(ks):
    index_pq = faiss.IndexPQ(D, k // 4, 4)
    index_pq.train(data)
    index_pq.add(data)
    found_pq = index_pq.search(quers, R)[1]
    recall_pq[i] = utils.mean_recall(found_pq, true)[0]

    print('k = %d' % k, 'Recall = %.3f' % recall_pq[i])

print(recall_pq)
plt.figure()
plt.semilogx(ks, recall_pq)
plt.savefig('./experiment58.png', bbox_inches='tight')
