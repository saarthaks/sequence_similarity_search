import numpy as np
import matplotlib.pyplot as plt
import torch
import faiss
from scipy.spatial.distance import pdist, squareform

from data_classes import BERTDataset

n = 600000
dim = 128
C = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
#C = [4]

file_q = './datasets/Q.pt'
file_db = './datasets/D.pt'
src = BERTDataset(file_q, file_db, n)
db = src.generate_db()
data = np.array(db)

intras = []
inters = []
ratios = []
for c in C:
    km = faiss.Kmeans(dim, c, niter=20, spherical=True)
    km.train(data)

    centroids = km.centroids
    _, I = km.index.search(data, 1)
    ordered_centroids = centroids[I[:,0]]
    dists = np.linalg.norm(data-ordered_centroids, axis=1)**2
    intra = dists.mean()
    intras.append(intra)

    dists = squareform(pdist(centroids))
    np.fill_diagonal(dists, np.inf)
    inter = dists.min()**2
    inters.append(inter)

    ratio = intra/inter
    ratios.append(ratio)

    print(c, ratio)

plt.figure(figsize=(8,15))
plt.subplot(311)
plt.semilogx(C, ratios)
plt.title('Ratio')

plt.subplot(312)
plt.semilogx(C, inters)
plt.title('Inter')

plt.subplot(313)
plt.semilogx(C, intras)
plt.title('Intra')

plt.savefig('./figures/experiment61.png', bbox_inches='tight')


