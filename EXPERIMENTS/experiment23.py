import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
import faiss
import lshfly

from data_classes import GLOVEDataset
from dist_perm import DistPerm
import utils

def main():

    n = 10000
    D = 300
    k = 64

    num_queries = 100
    R = 200
    ds = [8, 12, 16, 20, 24, 28, 32]

    file_db = './datasets/glove_6B_300d.pt'
    data_source = GLOVEDataset(file_db, D, n)
    db = data_source.generate_db()
    data = np.array(db).astype(np.float32)
    #data = data / np.linalg.norm(data, axis=1)[:, np.newaxis]
    queries = data_source.generate_queries(num_queries)
    quers = np.array(queries).astype(np.float32)
    #quers = quers / np.linalg.norm(quers, axis=1)[:, np.newaxis]
    
    #print(data.shape)
    #print(quers.shape)
    index_l2 = faiss.IndexFlatL2(D)
    index_l2.add(data)
    true = index_l2.search(quers, R)[1]
    
    MAPs_dp = len(ds)*[0]
    MAPs_lsh = len(ds)*[0]
    #MAPs2 = len(ds)*[0]
    #MAPs3 = len(ds)*[0]
    #MAPs4 = len(ds)*[0]
    for j, dj in enumerate(ds):
        index_dp = DistPerm(k, d=dj)
        #index_dp2 = DistPerm(k, d=dj)
        #index_dp3 = DistPerm(k, d=dj)
        index_lsh = faiss.IndexLSH(D, dj)
        index_fly = lshfly.flylsh(data, hash_length=dj, sampling_ratio=0.1, embedding_size=20*dj)
        # index_lsh = faiss.IndexPQ(D, dj, 8)

        index_dp.fit(db)
        index_dp.add(db)
        #index_dp2.fit(db, alg='fft')
        #index_dp2.add(db)
        #index_dp3.fit(db, alg='weighted')
        #index_dp3.add(db)

        index_lsh.train(data)
        index_lsh.add(data)
        # index_l2.add(np.array(db))

        found_dp = index_dp.search(queries, R).numpy()
        found_lsh = index_lsh.search(quers, R)[1]
        #found3 = index_dp2.search(queries, R).numpy()
        #found4 = index_dp3.search(queries, R).numpy()

        MAPs_dp[j] = utils.mean_avg_precision(found_dp, true)[0]
        MAPs_lsh[j] = utils.mean_avg_precision(found_lsh, true)[0]
        flymap = index_fly.findmAP(10000//50, n_points=100)
        #MAPs2[j] = np.mean(avg_precision2)
        #MAPs3[j] = np.mean(avg_precision3)
        #MAPs4[j] = np.mean(avg_precision4)
        
        print('AP@%d: %.3f [Random]' % (R, MAPs_dp[j]))
        print('AP@%d: %.3f [LSH]' % (R, MAPs_lsh[j]))
        print('AP@%d: %.3f [Fly]\n' % (R, flymap))

    #MAPs = np.array(MAPs)
    #MAPs2 = np.array(MAPs2)
    #MAPs3 = np.array(MAPs3)
    #MAPs4 = np.array(MAPs4)

    #plt.figure(figsize=(8,6))
    #plt.plot(ds, 100*MAPs, '--*k', label='Random')
    #plt.plot(ds, 100*MAPs3, '--^b', label='FFT')
    #plt.plot(ds, 100*MAPs4, '--xg', label='Weighted KMeans')
    #plt.plot(ds, 100*MAPs2, '--or', label='LSH')
    # plt.fill_between(Rs, 100*(means-2*stds), 100*(means+2*stds), color=(0.8, 0.8, 0.8))
    #plt.ylim([0, 100])
    #plt.ylabel('Mean Average Precision [%]')
    #plt.xlabel('Number of Anchors Used')
    #plt.title('BERT Dataset with %d Entries, Retrieving Top-%s, # Anchors (Total) = %d' % (n, R, k))
    #plt.legend()

    #plt.savefig('./figures/experiment23.png', bbox_inches='tight')

    print('\nDONE\n')

if __name__ == '__main__':
    main()
