import torch
import numpy as np

from data_classes import *

n = 10000
filename = './datasets/SIFT10Mfeatures.mat'
source = SIFTDataset(filename, n)
db = source.generate_db()

norms = torch.linalg.norm(db, axis=1)
print('SIFT Vec norms: %.3f +/- %.4f' % (norms.mean(), norms.std()))

file_q = './datasets/Q.pt'
file_db = './datasets/D.pt'
source = BERTDataset(file_q, file_db, n)
db = source.generate_db()

norms = torch.linalg.norm(db, axis=1)
print('BERT Vec norms: %.3f +/- %.4f' % (norms.mean(), norms.std()))
