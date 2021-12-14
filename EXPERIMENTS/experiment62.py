import numpy as np
import torch
import faiss

from data_classes import BERTDataset
from dist_perm import DistPerm
import utils

n = 300000
D = 128

m = 1024
k = 8

file_q = './datasets/Q.pt'
file_db = './datasets/D.pt'
src = BERTDataset(file_q, file_db, n)
db = src.generate_db()
data = np.array(db).astype(np.float32)

DP = DistPerm(m, k=k)
DP.fit(db, alg='kmeans')
ranks = DP.add(db)
ordering = ranks.argsort(axis=1)[:,:k].numpy()

A = np.zeros((n, m))
for i, order in enumerate(ordering):
    A[i, order] = 1

np.savez('./experiment62.npz', A=A)
print('DONE')
