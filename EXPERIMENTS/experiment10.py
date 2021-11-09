import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
import faiss

from data_classes import SIFTDataset
from dist_perm import DistPerm

def main():

    n = 10000
    D = 128
    k = 64

    num_queries = 100
    R = 200
    ds = [8, 12, 16, 20, 24, 28, 32]

    data_file = './datasets/SIFT10Mfeatures.mat'
    data_source = SIFTDataset(data_file, n)
    index_dp = DistPerm(k)
    index_dp2 = DistPerm(k)
    index_dp3 = DistPerm(k)

    db = data_source.generate_db()
    data = np.array(db).astype(np.float32)
    #means = np.array(data_source.db_means).astype(np.float32)
    #stds = np.array(data_source.db_stds).astype(np.float32)
    #data = (data - means[np.newaxis, :]) / stds[np.newaxis, :]

    MAPs = len(ds)*[0]
    MAPs2 = len(ds)*[0]
    MAPs3 = len(ds)*[0]
    MAPs4 = len(ds)*[0]
    for j, dj in enumerate(ds):
        index_lsh = faiss.IndexLSH(D, dj)
        index_l2 = faiss.IndexFlatL2(D)
        # index_lsh = faiss.IndexPQ(D, dj, 8)
        index_dp.d = dj
        index_dp2.d = dj
        index_dp3.d = dj

        index_dp.fit(db)
        index_dp.add(db)
        index_dp2.fit(db, alg='fft')
        index_dp2.add(db)
        index_dp3.fit(db, alg='weighted')
        index_dp3.add(db)


        index_lsh.train(data)
        index_lsh.add(data)
        index_l2.add(data)
        # index_l2.add(np.array(db))

        queries = data_source.generate_queries(num_queries)
        quers = np.array(queries).astype(np.float32)
        #quers = (quers - means[np.newaxis, :]) / stds[np.newaxis, :]
        found = index_dp.search(queries, R).numpy()
        found2 = index_lsh.search(quers, R)[1]
        found3 = index_dp2.search(queries, R).numpy()
        found4 = index_dp3.search(queries, R).numpy()
        true = index_l2.search(quers, R)[1]

        # precision_R = num_queries*[0]
        avg_precision = num_queries*[0]
        avg_precision2 = num_queries*[0]
        avg_precision3 = num_queries*[0]
        avg_precision4 = num_queries*[0]
        for i in range(num_queries):
            hits_vec = np.array([1*(f in true[i]) for f in found[i]])
            hits_vec2 = np.array([1*(f in true[i]) for f in found2[i]])
            hits_vec3 = np.array([1*(f in true[i]) for f in found3[i]])
            hits_vec4 = np.array([1*(f in true[i]) for f in found4[i]])

            precision_R = np.cumsum(hits_vec)/np.arange(1, R+1)
            precision_R2 = np.cumsum(hits_vec2)/np.arange(1, R+1)
            precision_R3 = np.cumsum(hits_vec3)/np.arange(1, R+1)
            precision_R4 = np.cumsum(hits_vec4)/np.arange(1, R+1)

            avg_precision[i] = sum(hits_vec*precision_R)/R
            avg_precision2[i] = sum(hits_vec2*precision_R2)/R
            avg_precision3[i] = sum(hits_vec3*precision_R3)/R
            avg_precision4[i] = sum(hits_vec4*precision_R4)/R

        MAPs[j] = np.mean(avg_precision)
        MAPs2[j] = np.mean(avg_precision2)
        MAPs3[j] = np.mean(avg_precision3)
        MAPs4[j] = np.mean(avg_precision4)
        print('AP@%d: %.3f +/- %.4f [Random]' % (R, np.mean(avg_precision), np.std(avg_precision)))
        print('AP@%d: %.3f +/- %.4f [FFT]' % (R, np.mean(avg_precision3), np.std(avg_precision3)))
        print('AP@%d: %.3f +/- %.4f [Weighted]' % (R, np.mean(avg_precision4), np.std(avg_precision4)))
        print('AP@%d: %.3f +/- %.4f [LSH]\n' % (R, np.mean(avg_precision2), np.std(avg_precision2)))

    MAPs = np.array(MAPs)
    MAPs2 = np.array(MAPs2)
    MAPs3 = np.array(MAPs3)
    MAPs4 = np.array(MAPs4)

    plt.figure(figsize=(8,6))
    plt.plot(ds, 100*MAPs, '--*k', label='Random')
    plt.plot(ds, 100*MAPs3, '--^b', label='FFT')
    plt.plot(ds, 100*MAPs4, '--xg', label='Weighted KMeans')
    plt.plot(ds, 100*MAPs2, '--or', label='LSH')
    # plt.fill_between(Rs, 100*(means-2*stds), 100*(means+2*stds), color=(0.8, 0.8, 0.8))
    plt.ylim([0, 100])
    plt.ylabel('Mean Average Precision [%]')
    plt.xlabel('Number of Anchors Used')
    plt.title('Dataset with %d Entries, Retrieving Top-%s, # Anchors (Total) = %d' % (n, R, k))
    plt.legend()

    plt.savefig('./figures/experiment10.png', bbox_inches='tight')

    print('\nDONE\n')

if __name__ == '__main__':
    main()
