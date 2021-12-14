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
    trials = 1
    num_queries = 100

    Rs = [50, 100, 200]
    R = max(Rs)
    ks = [8, 16, 48, 120]

    file_db = './datasets/glove_6B_300d.pt'
    data_source = GLOVEDataset(file_db, D, n)
    dbs = [data_source.generate_db() for _ in range(trials)]
    datas = [np.array(db).astype(np.float32) for db in dbs]
    queries = data_source.generate_queries(num_queries)
    quers = np.array(queries).astype(np.float32)
    
    trues = []
    for data in datas:
        index_l2 = faiss.IndexFlatL2(D)
        index_l2.add(data)
        trues.append(index_l2.search(quers, R)[1])
    
    MAPs_dp = np.zeros((trials, len(ks), len(Rs)))
    MAPs_knn = np.zeros_like(MAPs_dp)
    MAPs_fft = np.zeros_like(MAPs_dp)

    #MAPs2 = len(ds)*[0]
    #MAPs3 = len(ds)*[0]
    #MAPs4 = len(ds)*[0]
    for tri in range(trials):
        for i,k in enumerate(ks):
            index_dp = DistPerm(k**2//2, d=k)
            index_dp.fit(dbs[tri])
            index_dp.add(dbs[tri])
            found_dp = index_dp.search(queries, R).numpy()
            MAPs_dp[tri, i] = [utils.mean_avg_precision(found_dp[:,:r], trues[tri][:,:r])[0] for r in Rs]

            index_dp = DistPerm(k**2//2, d=k)
            index_dp.fit(dbs[tri], alg='weighted')
            found_dp = index_dp.search(queries, R).numpy()
            MAPs_knn[tri, i] = [utils.mean_avg_precision(found_dp[:,:r], trues[tri][:,:r])[0] for r in Rs]

            index_dp = DistPerm(k**2//2, d=k)
            index_dp.fit(dbs[tri], alg='fft')
            found_dp = index_dp.search(queries, R).numpy()
            MAPs_fft[tri, i] = [utils.mean_avg_precision(found_dp[:,:r], trues[tri][:,:r])[0] for r in Rs]

        if tri == 0:
            print(MAPs_dp[0])
            print(MAPs_knn[0])
            print(MAPs_fft[0])


    x_ticks = np.arange(0,2*len(ks),2)
    plt.figure(figsize=(8,15))
    plt.subplot(311)
    width = 0.3
    plt.bar(x_ticks - 0.45, np.mean(MAPs_dp[:,:,0], axis=0), width, yerr=np.std(MAPs_dp[:,:,0],axis=0), label='DP-Random')
    plt.bar(x_ticks - 0.15, np.mean(MAPs_fft[:,:,0], axis=0), width, yerr=np.std(MAPs_fft[:,:,0],axis=0), label='DP-FFT')
    plt.bar(x_ticks + 0.15, np.mean(MAPs_knn[:,:,0], axis=0), width, yerr=np.std(MAPs_knn[:,:,0],axis=0), label='DP-kNN')

    plt.xticks(x_ticks, ks)
    plt.xlabel('Num. of Signals')
    plt.ylabel('Mean Average Precision')
    plt.title('GLoVE Dataset (300-dim) with 10000 Entries, Retrieving Top-50 (0.5%)')
    plt.ylim([0,1])
    plt.legend()

    plt.subplot(312)
    width = 0.3
    plt.bar(x_ticks - 0.45, np.mean(MAPs_dp[:,:,1], axis=0), width, yerr=np.std(MAPs_dp[:,:,1],axis=0), label='DP-Random')
    plt.bar(x_ticks - 0.15, np.mean(MAPs_fft[:,:,1], axis=0), width, yerr=np.std(MAPs_fft[:,:,1],axis=0), label='DP-FFT')
    plt.bar(x_ticks + 0.15, np.mean(MAPs_knn[:,:,1], axis=0), width, yerr=np.std(MAPs_knn[:,:,1],axis=0), label='DP-kNN')
    plt.xticks(x_ticks, ks)
    plt.xlabel('Num. of Signals')
    plt.ylabel('Mean Average Precision')
    plt.title('GLoVE Dataset (300-dim) with 10000 Entries, Retrieving Top-100 (1%)')
    plt.ylim([0,1])
    plt.legend()

    plt.subplot(313)
    width = 0.3
    plt.bar(x_ticks - 0.45, np.mean(MAPs_dp[:,:,2], axis=0), width, yerr=np.std(MAPs_dp[:,:,2],axis=0), label='DP-Random')
    plt.bar(x_ticks - 0.15, np.mean(MAPs_fft[:,:,2], axis=0), width, yerr=np.std(MAPs_fft[:,:,2],axis=0), label='DP-FFT')
    plt.bar(x_ticks + 0.15, np.mean(MAPs_knn[:,:,2], axis=0), width, yerr=np.std(MAPs_knn[:,:,2],axis=0), label='DP-kNN')
    plt.xticks(x_ticks, ks)
    plt.xlabel('Num. of Signals')
    plt.ylabel('Mean Average Precision')
    plt.title('GLoVE Dataset (300-dim) with 10000 Entries, Retrieving Top-200 (2%)')
    plt.ylim([0,1])
    plt.legend()

    plt.savefig('./figures/experiment32.png', bbox_inches='tight')



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
