import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import NearestNeighbors
import faiss

from lshfly import flylsh 
from data_classes import GLOVEDataset
from dist_perm import DistPerm
import utils

def main():

    n = 10000
    #frac = [2e-3, 1e-2, 2e-2, 5e-2, 1e-1]
    D = 300
    ks = [2, 4, 8, 16, 32, 64, 128, 256]
    R = 200
    num_queries = 100

    file_db = './datasets/glove_6B_300d.pt'
    data_source = GLOVEDataset(file_db, D, n)
    db = data_source.generate_db()
    data = np.array(db).astype(np.float32)
    queries = data_source.generate_queries(num_queries)
    quers = np.array(queries).astype(np.float32)
    
    index_l2 = faiss.IndexFlatL2(D)
    index_l2.add(data)
    true = index_l2.search(quers, R)[1]
    
    APs = np.zeros((2,len(ks)))
    for j, k in enumerate(ks):
        index_dp = DistPerm(k)

        index_dp.fit(db)
        index_dp.add(db)
        found_dp = index_dp.search(queries,R).numpy()
        APs[:,j] = utils.mean_avg_precision(found_dp, true)

        print('AP@200: %.3f +/- %.4f [Random]' % (APs[0,j], APs[1,j]))

    print('\nDONE\n')

    plt.figure(figsize=(8,6))
    plt.errorbar(ks, 100*APs[0], yerr=100*APs[1], fmt='--*k', capsize=4)
    plt.semilogx()
    plt.xlabel('Num. of Anchors')
    plt.ylabel('Mean Average Precision [%]')

    plt.savefig('./figures/experiment25.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
