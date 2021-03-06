import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
import faiss
from fly import fly

from data_classes import BERTDataset
from dist_perm import DistPerm
import utils

def main():

    n = 300000
    D = 128
    trials = 5
    num_queries = 100

    Rs = [50, 100, 200]
    R = max(Rs)
    ks = [8, 16, 32, 64]

    file_q = './datasets/Q.pt'
    file_db = './datasets/D.pt'
    data_source = BERTDataset(file_q, file_db, n)
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
    MAPs_pq = np.zeros_like(MAPs_dp)
    MAPs_lsh = np.zeros_like(MAPs_dp)
    MAPs_fly = np.zeros_like(MAPs_dp)

    #MAPs2 = len(ds)*[0]
    #MAPs3 = len(ds)*[0]
    #MAPs4 = len(ds)*[0]
    for tri in range(trials):
        for i,k in enumerate(ks):
            index_dp = DistPerm(20*k, k=k)
            index_dp.fit(dbs[tri])
            index_dp.add(dbs[tri])
            found_dp = index_dp.search(queries, R).numpy()
            MAPs_dp[tri, i] = [utils.mean_recall(found_dp[:,:r], trues[tri][:,:r]) for r in Rs]

            index_pq = faiss.IndexPQ(D, k // 4, 4)
            index_pq.train(datas[tri])
            index_pq.add(datas[tri])
            found_pq = index_pq.search(quers, R)[1]
            MAPs_pq[tri, i] = [utils.mean_recall(found_pq[:,:r], trues[tri][:,:r]) for r in Rs]

            index_lsh = faiss.IndexLSH(D, k)
            index_lsh.train(datas[tri])
            index_lsh.add(datas[tri])
            found_lsh = index_lsh.search(quers, R)[1]
            MAPs_lsh[tri, i] = [utils.mean_recall(found_lsh[:,:r], trues[tri][:,:r]) for r in Rs]

            index_fly = fly(k, 20*k)
            index_fly.fit(datas[tri], sampling_ratio=0.1)
            index_fly.add(datas[tri])
            found_fly = index_fly.search(quers, R)
            MAPs_fly[tri, i] = [utils.mean_recall(found_fly[:,:r], trues[tri][:,:r]) for r in Rs]

        if tri == 0:
            print(MAPs_dp[0])
            print(MAPs_pq[0])
            print(MAPs_lsh[0])
            print(MAPs_fly[0])


    x_ticks = np.arange(0,2*len(ks),2)
    plt.figure(figsize=(8,15))
    plt.subplot(311)
    width = 0.3
    plt.bar(x_ticks - 0.45, np.mean(MAPs_lsh[:,:,0], axis=0), width, yerr=np.std(MAPs_lsh[:,:,0],axis=0), label='LSH')
    plt.bar(x_ticks - 0.15, np.mean(MAPs_pq[:,:,0], axis=0), width, yerr=np.std(MAPs_pq[:,:,0],axis=0), label='PQ')
    plt.bar(x_ticks + 0.15, np.mean(MAPs_fly[:,:,0], axis=0), width, yerr=np.std(MAPs_fly[:,:,0],axis=0), label='Fly')
    plt.bar(x_ticks + 0.45, np.mean(MAPs_dp[:,:,0], axis=0), width, yerr=np.std(MAPs_dp[:,:,0],axis=0), label='DP')
    plt.xticks(x_ticks, ks)
    plt.xlabel('Num. of Signals')
    plt.ylabel('Mean Recall')
    plt.title('BERT Dataset (128-dim) with 300,000 Entries, Retrieving Top-50 (0.016%)')
    plt.ylim([0, 1])
    plt.legend()

    plt.subplot(312)
    width = 0.3
    plt.bar(x_ticks - 0.45, np.mean(MAPs_lsh[:,:,1], axis=0), width, yerr=np.std(MAPs_lsh[:,:,1],axis=0), label='LSH')
    plt.bar(x_ticks - 0.15, np.mean(MAPs_pq[:,:,1], axis=0), width, yerr=np.std(MAPs_pq[:,:,1],axis=0), label='PQ')
    plt.bar(x_ticks + 0.15, np.mean(MAPs_fly[:,:,1], axis=0), width, yerr=np.std(MAPs_fly[:,:,1],axis=0), label='Fly')
    plt.bar(x_ticks + 0.45, np.mean(MAPs_dp[:,:,1], axis=0), width, yerr=np.std(MAPs_dp[:,:,1],axis=0), label='DP')
    plt.xticks(x_ticks, ks)
    plt.xlabel('Num. of Signals')
    plt.ylabel('Mean Recall')
    plt.title('BERT Dataset (128-dim) with 300,000 Entries, Retrieving Top-100 (0.033%)')
    plt.ylim([0, 1])
    plt.legend()

    plt.subplot(313)
    width = 0.3
    plt.bar(x_ticks - 0.45, np.mean(MAPs_lsh[:,:,2], axis=0), width, yerr=np.std(MAPs_lsh[:,:,2],axis=0), label='LSH')
    plt.bar(x_ticks - 0.15, np.mean(MAPs_pq[:,:,2], axis=0), width, yerr=np.std(MAPs_pq[:,:,2],axis=0), label='PQ')
    plt.bar(x_ticks + 0.15, np.mean(MAPs_fly[:,:,2], axis=0), width, yerr=np.std(MAPs_fly[:,:,2],axis=0), label='Fly')
    plt.bar(x_ticks + 0.45, np.mean(MAPs_dp[:,:,2], axis=0), width, yerr=np.std(MAPs_dp[:,:,2],axis=0), label='DP')
    plt.xticks(x_ticks, ks)
    plt.xlabel('Num. of Signals')
    plt.ylabel('Mean Recall')
    plt.title('BERT Dataset (128-dim) with 300,000 Entries, Retrieving Top-200 (0.066%)')
    plt.ylim([0, 1])
    plt.legend()

    plt.savefig('./figures/experiment49.png', bbox_inches='tight')



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
