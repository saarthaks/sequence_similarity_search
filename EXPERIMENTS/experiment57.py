import numpy as np
import torch
import faiss

from data_classes import BERTDataset
from dist_perm import DistPerm
import utils

n = 600000  
D = 128     
num_q = 5   
m = 300     
k = 4       
R = 200
        
file_q = './datasets/Q.pt'
file_db = './datasets/D.pt'
src = BERTDataset(file_q, file_db, n)
db = src.generate_db()
data = np.array(db).astype(np.float32)
query = src.generate_queries(num_q)
quers = np.array(query).astype(np.float32)
    
DP = DistPerm(m, k)
DP.fit(db, 'kmeans')
DP.add(db)
#kmeans = faiss.Kmeans(D, m, niter=20, spherical=True)
#kmeans.train(data)
#DP.anchors = torch.from_numpy(kmeans.centroids)
_,I = DP.km.index.search(quers, 1)
print(I)
found, dists = DP.search(query, R)
found = found.numpy()
print(found[0])
for q,f,di in zip(quers, found, dists):
    #_, I = DP.km.index.search(q, 1)
    print('Query is in cluster %s' % str(I[0,0]))
    vecs = data[f]
    perms = DP.rank_index[vecs]
    _, I = DP.km.index.search(vecs, 1)
    cs = I[:,0]
    print(cs)
    print(di)
    print()

