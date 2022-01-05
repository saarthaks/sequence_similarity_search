import numpy as np
import matplotlib.pyplot as plt
import torch
import faiss
import sys
# np.random.seed(1235)

# import lshfly 
sys.path.insert(1, '/Users/derekhuang/Documents/Research/sequence_similarity_search/classes/')
# sys.path.insert(0, '/Users/derekhuang/Documents/Research/sequence_similarity_search/classes/data_classes.py')
# sys.path.insert(0, '/Users/derekhuang/Documents/Research/sequence_similarity_search/classes/lsh.py')
# sys.path.insert(0, '/Users/derekhuang/Documents/Research/sequence_similarity_search/classes/dist_perm.py')
from sklearn.neighbors import NearestNeighbors

from data_classes import BERTDataset
from dist_perm import DistPerm
from fly import Fly
import utils

def main():
    
    n = 10000
    D = 128
    num_queries = 1
    R = 100
    k = 2

    num_anchs = [8, 16, 32, 64, 128, 256, 512]

    file_q = './datasets/Q.pt'
    file_db = './datasets/D.pt'
    data_source = BERTDataset(file_q, file_db, n)
    db = data_source.generate_db()
    data = np.array(db).astype(np.float32)
    queries = data_source.generate_queries(num_queries)
    quers = np.array(queries).astype(np.float32)
    # index_l2 = faiss.IndexFlatL2(D)
    # index_l2.add(data)
    # true = index_l2.search(quers, R)[1]
    index_l2 = NearestNeighbors()
    index_l2.fit(data)
    true = index_l2.kneighbors(quers, n_neighbors=R)[1]

    MAPs_dp = np.zeros_like(num_anchs, dtype=np.float32)
    MAPs_fly = np.zeros_like(num_anchs, dtype=np.float32)
    for i, num in enumerate(num_anchs):
        index_fly = Fly(k=k, m=num)
        fly_weights = index_fly.fit(db,sampling_ratio=.1)
        fly_hashes = index_fly.add(db)
        # index_fly = lshfly.flylsh(data, hash_length=k, sampling_ratio=0.1, embedding_size=num)
        # index_fly.queries = quers-np.mean(quers,axis=0)[None,:]
        # index_fly.queries_hashes = (index_fly.queries@index_fly.weights)>0
        # MAPs_fly[i] = index_fly.findmAP3(nnn=R, n_points=num_queries)
        # MAPs_fly[i] = index_fly.findmAP3(nnn=R)
        # print(index_fly.data.shape[0])
        index_dp = DistPerm(num, k=k, dist='dot')
        index_dp.fit(db, alg='fly', fly=fly_weights)
        # compare this to index_fly.hashes
        dp_hashes = index_dp.add(db)
        # difference = np.sum(np.abs(np.where(dp_hashes >= k, 0, 1) - fly_hashes))
        # print(difference)/
        dp_to_fly = np.where(dp_hashes >= k, 0, 1)
        # difference = np.abs(np.where(dp_hashes >= k - .5, 0, 1) - index_fly.hashes)
        # print(np.sum(difference))
        found_dp, query_ranks, dp_dist = index_dp.search(queries, R)
        # print(query_ranks.shape)
        # print(found_dp.shape)
        found_dp = found_dp.numpy()


        MAPs_dp[i] = utils.mean_avg_precision(found_dp, true)[0]

        found_fly, qhashes, fly_dist = index_fly.search(queries, R, hash=dp_to_fly)
        found_fly = found_fly.numpy()
        MAPs_fly[i] = utils.mean_avg_precision(found_fly, true)[0]



        fly_0 = np.argwhere(fly_dist.squeeze() == 0)[0].flatten()
        dp_0 = np.argwhere(dp_dist.squeeze() == 0)[0].flatten()
        fly_elems = found_fly.ravel()[fly_0]
        dp_elems = found_dp.ravel()[dp_0]

        if np.isscalar(dp_elems):
            dp_elems = [dp_elems]
        if np.isscalar(fly_elems):
            fly_elems = [fly_elems]
        if not len(fly_elems) and not len(dp_elems):
            print(True)
            print("Fly elements: {}".format(len(fly_elems)))
            print("DP elements: {}".format(len(dp_elems)))
        else:
            output = np.in1d(fly_elems, dp_elems).any()
            if not output:
                if len(fly_elems) and not len(dp_elems):
                    print(True)
                    print("Fly elements: {}".format(len(fly_elems)))
                    print("DP elements: {}".format(len(dp_elems)))
                else:
                    print(False)
                    print(dp_elems)
                    print(fly_elems)
            else:
                print(output)
                print("Fly elements: {}".format(len(fly_elems)))
                print("DP elements: {}".format(len(dp_elems)))



            # print(np.in1d(found_dp.ravel()[dp_0], found_fly.ravel()[fly_0]).any())
            # print(found_fly.flatten())
            # print(found_dp.flatten())
            # print(set(found_fly.flatten()[fly_0]) >= set(found_dp.flatten()[dp_0]))
        # difference = np.sum(np.abs(np.where(query_ranks >= k, 0, 1) - qhashes))
        # print(np.where(query_ranks >= k, 0, 1))
        # print(qhashes)
        # print(query_ranks[0].numpy())
        # print(qhashes)
        # difference = np.sum(np.abs(query_ranks[0].numpy() - qhashes[0].numpy()))
        # print(difference)
        # print(found_fly)
        # print(found_dp)
        # return
        # print(difference)
        # print(found_dp)
        # print(np.sum(found_fly - found_dp))
        # print(torch.sum(torch.subtract(found_fly, found_dp)))

        print('Num anchs: %d' % num, 'MAP [DP]: %.3f' % MAPs_dp[i], 'MAP [Fly]: %.3f' % MAPs_fly[i])

    plt.figure(figsize=(8,5))
    plt.plot(num_anchs, MAPs_dp, '--*', label='DP')
    plt.plot(num_anchs, MAPs_fly, '--*', label='Fly')
    plt.xlabel('Total number of anchors [k]')
    plt.ylabel('Mean average precision')
    plt.title('BERT Dataset, N = 10000, k = 8')
    plt.legend()
    plt.savefig('./figures/experiment38.png', bbox_inches='tight')
    print('\nDone\n')

if __name__ == '__main__':
    main()
