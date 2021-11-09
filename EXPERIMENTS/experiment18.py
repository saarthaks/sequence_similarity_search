import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import faiss

from data_classes import BERTDataset

n = 10000
D = 128
k = 128
R = 200

file_q = './datasets/Q.pt'
file_db = './datasets/D.pt'
source = BERTDataset(file_q, file_db, n)

index_l2 = faiss.IndexScalarQuantizer(D, faiss.ScalarQuantizer.QT_fp16)
index_nn = NearestNeighbors(n_neighbors=k, metric='euclidean')

db = source.generate_db()
data = np.array(db).astype(np.float16)
index_l2.add(data)
index_nn.fit(data)

q = np.array(source.generate_queries(1)).astype(np.float16)

dist_l2, found_l2 = index_l2.search(q, R)
dist_nn, found_nn = index_nn.kneighbors(q, n_neighbors=R)

#print(dist_l2, dist_nn)
#print(found_l2, found_nn)
inc = np.argwhere(found_l2.squeeze() - found_nn.squeeze())
if len(inc):
    print(inc)
    inc = inc.squeeze()
    print(inc)
    print(found_l2[0][inc])
    print(found_nn[0][inc])
#print(np.argwhere(found_l2.squeeze() != found_nn.squeeze()))

