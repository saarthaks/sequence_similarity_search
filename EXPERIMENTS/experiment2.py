import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
# import faiss

from data_classes import SubspaceDataset
from dist_perm import DistPerm

def main():

    n = 1000
    d = 8
    D = 128
    k = 16

    data_source = SubspaceDataset(d, D)
    index_dp = DistPerm(k)
    # index_l2 = faiss.IndexFlatL2(D)
    index_l2 = NearestNeighbors(n_neighbors=k)

    db = data_source.generate_db(n)
    index_dp.fit(db)
    index_dp.add(db)
    data = np.array(db)
    index_l2.fit(data)
    # index_l2.add(np.array(db))

    Rs = [5, 10, 20, 40, 80, 160]
    means = len(Rs)*[0]
    stds = len(Rs)*[0]
    num_queries = 20

    queries = data_source.generate_queries(num_queries)[0]
    found = index_dp.search(queries, max(Rs)).numpy()
    # true = index_l2.search(queries, k)
    true = index_l2.kneighbors(queries, n_neighbors=max(Rs))[1]

    for j, R in enumerate(Rs):
        recall = num_queries*[0]
        for i in range(num_queries):
            recall[i] = sum([1*(f in true[i][:R]) for f in found[i][:R]])/R
        means[j] = np.mean(recall)
        stds[j] = np.std(recall)

        print('Recall@%d: %.3f +/- %.4f' % (R, np.mean(recall), np.std(recall)))

    means = np.array(means)
    stds = np.mean(stds)
    plt.figure(figsize=(8,6))
    plt.plot(Rs, 100*means, '--*k')
    plt.fill_between(Rs, 100*(means-2*stds), 100*(means+2*stds), color=(0.8, 0.8, 0.8))
    plt.ylim([0, 100])
    plt.ylabel('Recall@R [%]')
    plt.xlabel('R')
    plt.title('Dataset with %d Entries, # Anchors = %d' % (n, k))

    plt.savefig('./figures/experiment2.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
