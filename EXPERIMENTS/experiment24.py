import numpy as np
import torch
import matplotlib.pyplot as plt

from data_classes import *
from utils import *

def main():

    # ID of SIFT dataset
    n = 10000
    data_file = './datasets/SIFT10Mfeatures.mat'
    #data_source = SIFTDataset(data_file, n)
    #db = data_source.generate_db().numpy()

    #ID_results = id_mle_estimator(db, k_list=list(range(4,40,4)))
    #x_dom = list(ID_results.keys())
    #y = list(ID_results.values())
    #print(y[-5:])
    plt.figure(figsize=(8,6))
    #plt.plot(x_dom, y)
    plt.xlabel('k (kNN)')
    plt.ylabel('ID')
    plt.title('Local Intrinsic Dim. Estimate for SIFT10M, N=%d' % n)
    #plt.savefig('./figures/SIFT_ID.png', bbox_inches='tight')

    # ID of BERT dataset
    n = 100000
    file_q = './datasets/Q.pt'
    file_db = './datasets/D.pt'
    data_source = BERTDataset(file_q, file_db, n)
    db = data_source.generate_db().numpy()

    k_list = list(2**np.arange(2,9))
    k_list.append(384)
    ID_results = id_mle_estimator(db, k_list=k_list)
    x_dom = list(ID_results.keys())
    y = list(ID_results.values())
    print(y[-5:])
    plt.figure(figsize=(8,6))
    plt.plot(x_dom, y, '--*')
    plt.xlabel('k (kNN)')
    plt.ylabel('ID')
    plt.semilogx()
    plt.title('Local Intrinsic Dim. Estimate for BERT Embeddings, N=%d' % n)
    plt.savefig('./figures/BERT_ID4.png', bbox_inches='tight')

    # ID of GLoVE dataset
    n = 10000
    file_db = './datasets/glove_6B_300d.pt'
    #data_source = GLOVEDataset(file_db, 300, n)
    #db = data_source.generate_db().numpy()

    #ID_results = id_mle_estimator(db, k_list=list(2**np.arange(2,9)))
    #x_dom = list(ID_results.keys())
    #y = list(ID_results.values())
    #print(y[-5:])
    plt.figure(figsize=(8,6))
    #plt.plot(x_dom, y)
    plt.xlabel('k (kNN)')
    plt.ylabel('ID')
    plt.title('Local Intrinsic Dim. Estimate for GLoVE Embeddings, N=%d' % n)
    #plt.savefig('./figures/glove_ID.png', bbox_inches='tight')
    return

if __name__ == '__main__':
    main()
