import numpy as np
import torch
import matplotlib.pyplot as plt

from data_classes import *
from utils import *

def main():

    k_list = 2**np.arange(2, 9)

    # ID of SIFT dataset
    n = 10000
    data_file = './datasets/SIFT10Mfeatures.mat'
    data_source = SIFTDataset(data_file, n)
    db = data_source.generate_db().numpy()

    ID_results = id_mle_estimator(db, k_list=k_list)
    x_dom = list(ID_results.keys())
    y = list(ID_results.values())
    print(y[-5:])
    plt.figure(figsize=(8,6))
    plt.plot(x_dom, y, '--*', label='SIFT')

    # ID of BERT dataset
    n = 10000
    file_q = './datasets/Q.pt'
    file_db = './datasets/D.pt'
    data_source = BERTDataset(file_q, file_db, n)
    db = data_source.generate_db().numpy()

    ID_results = id_mle_estimator(db, k_list=k_list)
    x_dom = list(ID_results.keys())
    y = list(ID_results.values())
    print(y[-5:])
    plt.plot(x_dom, y, '--*', label='BERT')

    # ID of GLoVE dataset
    n = 10000
    file_db = './datasets/glove_6B_300d.pt'
    data_source = GLOVEDataset(file_db, 300, n)
    db = data_source.generate_db().numpy()

    ID_results = id_mle_estimator(db, k_list=k_list)
    x_dom = list(ID_results.keys())
    y = list(ID_results.values())
    print(y[-5:])
    plt.plot(x_dom, y, '--*', label='GLoVE')
    plt.semilogx()
    plt.xlabel('k (kNN)')
    plt.ylabel('ID')
    plt.title('Local Intrinsic Dim. Estimate, N=%d' % n)
    plt.legend()
    plt.savefig('./figures/ID_comparison.png', bbox_inches='tight')
    return

if __name__ == '__main__':
    main()
