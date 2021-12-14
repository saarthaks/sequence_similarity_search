import numpy as np
import matplotlib.pyplot as plt
import torch
import faiss

from fly import fly
from data_classes import BERTDataset
from dist_perm import DistPerm
import utils

def main():
    
    n = 300000
    D = 128
    num_queries = 50
    R = 50
    ks = [4,6,8,12,16,24,32]
    ms = np.round(np.logspace(5,10,6,base=2)).astype(int)
    print(ms)


    file_q = './datasets/Q.pt'
    file_db = './datasets/D.pt'
    data_source = BERTDataset(file_q, file_db, n)
    db = data_source.generate_db()
    data = np.array(db).astype(np.float32)
    queries = data_source.generate_queries(num_queries)
    quers = np.array(queries).astype(np.float32)

    index_l2 = faiss.IndexFlatL2(D)
    index_l2.add(data)
    true = index_l2.search(quers, R)[1]
    
    plt.figure(figsize=(8,5))
    MRs_dp = np.zeros((len(ms),len(ks)))
    MRs_fly = np.zeros_like(MRs_dp)
    for j,m in enumerate(ms):
        index_dp = DistPerm(m, k=max(ks))
        index_dp.fit(db, alg='kmeans')

        for i, k in enumerate(ks):
            index_dp.k = k
            index_dp.add(db)
            found_dp = index_dp.search(queries, R).numpy()
            MRs_dp[j,i] = utils.mean_recall(found_dp, true)

            index_fly = fly(k,m)
            index_fly.fit(data, sampling_ratio=0.1)
            index_fly.add(data)
            found_fly = index_fly.search(quers, R)
            MRs_fly[j,i] = utils.mean_recall(found_fly, true)


            print('k=%d' % k, 'Num anchs: %d' % m, 'MR [DP-Km]: %.3f' % MRs_dp[j,i], 'MR [Fly]: %.3f' % MRs_fly[j,i])

    for i,k in enumerate(ks):
        plt.plot(ms, MRs_dp[:,i], '-*', label='DP-Km, k=%d'%k, color=plt.cm.Reds((i+1)/(len(ks)+1)))
        plt.plot(ms, MRs_fly[:,i], '--*', label='Fly, k=%d'%k, color=plt.cm.Reds((i+1)/(len(ks)+1)))
    
    plt.xlabel('Total number of anchors')
    plt.ylabel('Mean Recall')
    plt.ylim([0,1])
    plt.xscale('log')
    plt.title('BERT Dataset, N = 300000')
    plt.legend()
    plt.savefig('./figures/experiment52.png', bbox_inches='tight')
    np.savez('./experiment52_data.npz', dp=MRs_dp, fly=MRs_fly)
    print('\nDone\n')

if __name__ == '__main__':
    main()
