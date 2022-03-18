# -*- coding: utf-8 -*-
"""Copy of diffsort.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aQB8NiOAMFp9_Ej29vUMiO6fY8W8hO4i
"""

# !gdown --id 1W3PsVmRjs1N4yBK0uZBXZbedJRQnaBw8
# !gdown --id 1Ci1NlbnzLNrscBxD3hdpwPE7cBkjuqkk

from google.colab import drive
drive.mount('/content/gdrive')

# !rm -rf sequence_similarity_search
!git clone https://github.com/saarthaks/sequence_similarity_search.git

!git clone https://github.com/google-research/fast-soft-sort.git

!python3 fast-soft-sort/setup.py install

!pip3 install torchsort
!pip3 install faiss-cpu

!pwd
!ls

import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
import sys
sys.path.insert(1, './sequence_similarity_search/classes')
sys.path.insert(1, './fast-soft-sort/fast_soft_sort')
# sys.path.insert(1, '/Users/derekhuang/Documents/Research/fast-soft-sort/fast_soft_sort')
from data_classes import BERTDataset
from dist_perm import DistPerm
import utils
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torchsort

def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # console = logging.StreamHandler()
    # console.setFormatter(log_formatter)
    # logger.addHandler(console)


def soft_rank(array):
  #  return pytorch_ops.soft_rank(array.cpu(), direction="DESCENDING", regularization_strength=.001).cuda()
   return torchsort.soft_rank(-1 * array, regularization_strength=.000001)

class AnchorNet(nn.Module):
    def __init__(self, num_anchs, d, k, hidden=128, hidden2=128, out=128):
        super(AnchorNet, self).__init__()
        self.transform = nn.Sequential(
          # nn.Linear(d, hidden),
          # nn.ReLU(),
          # nn.Linear(hidden, hidden2),
          # nn.ReLU(),
          # nn.Linear(hidden, hidden2),
          # nn.Linear(hidden2, out)
        )
        self.anchors = nn.Linear(out, num_anchs)
        self.k = k

    def forward(self, data, query):
        # data_out = self.transform(data)
        # query_out = self.transform(query)
        # data_rank = torch.clamp(soft_rank(self.anchors(data_out)), max=k)
        # query_rank = torch.clamp(soft_rank(self.anchors(query_out)), max=k)
        # data_rank = soft_rank(self.anchors(data))
        # query_rank = soft_rank(self.anchors(query))
        data_rank = self.anchors(data)
        query_rank = self.anchors(query)
        anchor_norm = torch.norm(self.anchors.weight, dim=0)
        data_rank = soft_rank(torch.div(data_rank, anchor_norm))
        query_rank = soft_rank(torch.div(query_rank, anchor_norm))
        out = torch.matmul(query_rank, data_rank.T)
        out = torch.clamp(soft_rank(out), max=k)
        return out

    def evaluate(self, data, query):
        data_rank = self.anchors(data)
        query_rank = self.anchors(query)
        anchor_norm = torch.norm(self.anchors.weight, dim=0)
        d_dist = torch.div(data_rank, anchor_norm)
        q_dist = torch.div(query_rank, anchor_norm)

        query_ranks = self.k*torch.ones(q_dist.shape, dtype=torch.float)
        data_ranks = self.k*torch.ones(d_dist.shape, dtype=torch.float)

        query_ids = torch.argsort(q_dist, dim=1)[:, :self.k]
        data_ids = torch.argsort(d_dist, dim=1)[:, :self.k]

        q_ids = torch.arange(query.shape[0])[:,None]
        d_ids = torch.arange(data.shape[0])[:,None]
        query_ranks[q_ids, query_ids] = torch.arange(self.k, dtype=torch.float)
        data_ranks[d_ids, data_ids] = torch.arange(self.k, dtype=torch.float)

        db_dists = torch.cdist(data_ranks, query_ranks, p=1).float()
        closest_idx = torch.topk(db_dists, k, dim=0, largest=False)
        return closest_idx[1].transpose(0,1), query_ranks, closest_idx[0]
        # return closest_idx, query_ranks, closest_idx[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

n = 200000
D = 128
num_queries = 3200
num_anchors = 128
R = 100
k = 32

file_q = './Q.pt'
file_db = './D.pt'
data_source = BERTDataset(file_q, file_db, n)
db = data_source.generate_db()
data = np.array(db).astype(np.float32)
queries = data_source.generate_queries(num_queries)
quers = np.array(queries).astype(np.float32)

logger = logging.getLogger()
setup_logger(logger)
logging.basicConfig(filename='example.log', level=logging.DEBUG)

import torch, itertools
from torch.utils.data import TensorDataset, random_split, DataLoader
from sklearn.model_selection import train_test_split

def dataset_split(dataset, train_frac):
    length = len(dataset)
   
    train_length = int(length * train_frac)
    valid_length = int((length - train_length) / 2)
    test_length  = length - train_length - valid_length

    sets = random_split(dataset, (train_length, valid_length, test_length))
    dataset = {name: set for name, set in zip(('train', 'val', 'test'), sets)}
    return dataset

# Fits fine in mem
batch_size=3200
train_split = .8

query_datasets = dataset_split(quers, train_split)
# doc_datasets = dataset_split(db, train_split)
doc_datasets = dataset_split(db, train_split)

# The fixed train queries
query_data_loader = torch.utils.data.DataLoader(dataset=query_datasets['train'], 
                                           batch_size=batch_size, 
                                           shuffle=False)

# The fixed test queries 
query_data_test_loader = torch.utils.data.DataLoader(dataset=query_datasets['test'], 
                                          batch_size=batch_size, 
                                          shuffle=False)

# The fixed val queries 
query_data_val_loader = torch.utils.data.DataLoader(dataset=query_datasets['val'], 
                                          batch_size=batch_size, 
                                          shuffle=False)

# The train docs
docs_loader = torch.utils.data.DataLoader(dataset=doc_datasets['train'], 
                                           batch_size=5000, 
                                           shuffle=False)

# The test docs
docs_test_loader = torch.utils.data.DataLoader(dataset=doc_datasets['test'], 
                                           batch_size=5000, 
                                           shuffle=False)

docs_val_loader = torch.utils.data.DataLoader(dataset=doc_datasets['val'], 
                                           batch_size=5000, 
                                           shuffle=False)

# Generate the nearest neighbors for each batch of docs 
# d = docs
# ret = train/test
def return_loader(d, ret):
    index_l2 = NearestNeighbors()
    index_l2.fit(d)
    q = None
#   Get the right data
    if ret=='query':
        q = next(iter(query_data_loader))
    elif ret=='test':
        q = next(iter(query_data_test_loader))
    elif ret=='val':
        q = next(iter(query_data_val_loader))
#   Get the true nearest neighbors
    true = torch.tensor(index_l2.kneighbors(q, n_neighbors=k)[1])
    # print(true)÷
    query_data = []
    for i in range(q.shape[0]):
        ground_truth = np.arange(len(d)) + 1
        top = np.arange(k) + 1
        # print(len(list(true[i].view(-1, 1))))
        # print(len(top))
        ground_truth.put(true[i].view(-1, 1), top)
        query_data.append([q[i], ground_truth])
    test_set = dataset_split(query_data, 1)
#   Return data as a dataloader
    data = torch.utils.data.DataLoader(dataset=test_set['train'], 
                                           batch_size=320, 
                                           shuffle=False)
    return data

model = AnchorNet(num_anchors, D, k).to(device)
criterion = nn.MSELoss()
lr=.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

with torch.no_grad():
    correct = 0
    total = 0
    for d in docs_test_loader:
    # for d in docs_loader:
        test_loader = return_loader(d, ret='test')
        d = d.to(device)
        for q, l in test_loader:
#             print(q)
            q = q.to(device)
            l = l.to(device)
            outputs = model.evaluate(d,q)
#           Recall: TP / (TP + TN)
            predicted = outputs[0][:,2]
            # print(predicted)
            # print(predicted./shape)
            # total += l.size(0)
            # correct += (predicted.to(device) == l.flatten()).sum().item()

    # print ('recall_test: {:.4f}'
            # .format(correct / total))

loss_steps = []
for epoch in range(150):
#   Train step: get batch of train documents, then generate the dataloader containing 
#   the train queries and the correct data labels
    train_correct = 1
    train_total = 1
    for step, d in enumerate(docs_loader):
        query_loader = return_loader(d, ret='query')
        d = d.to(device)
        for i, (q, l) in enumerate(query_loader):  
            q = q.to(device)
            l = l.to(device)
            outputs = model(d,q)
            loss = criterion(outputs, l.squeeze().float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            # train_total += l.size(0)
            # train_correct += (predicted == l.flatten()).sum().item()
        # print('Epoch [{}] Step [{}] Loss: {:.4f}'.format(epoch, step, loss.item()))
        
#   Eval step: repeat but with the test documents and test queries      
    if epoch % 1 == 0:
        with torch.no_grad():
            correct = 1
            total = 1
            correct_val = 1
            total_val = 1
            for d in docs_test_loader:
            # for d in docs_loader:
                test_loader = return_loader(d, ret='test')
                d = d.to(device)
                for q, l in test_loader:
                    q = q.to(device)
                    l = l.to(device)
                    lol = model(d,q)
                    test_loss = criterion(lol, l.squeeze())
                    outputs = model.evaluate(d,q)
        #           Recall: TP / (TP + TN)
                    predicted = outputs[0][:,0]
                    # total += l.size(0)
                    # correct += (predicted.to(device) == l.flatten()).sum().item()


            for d in docs_val_loader:
            # for d in docs_loader:
                test_loader = return_loader(d, ret='val')
                d = d.to(device)
                for q, l in test_loader:
                    q = q.to(device)
                    l = l.to(device)
                    outputs = model.evaluate(d,q)
        #           Recall: TP / (TP + TN)
                    predicted = outputs[0][:,0]
                    # total_val += l.size(0)
                    # correct_val += (predicted.to(device) == l.flatten()).sum().item()

            # loss_steps.append(loss.item())
            # plt.plot(loss_steps)
            # plt.savefig('test.png')

            print ('Epoch [{}], Loss: {:.4f}, Test Loss: {:.4f}, recall_train: {:.4f}, recall_test: {:.4f} recall_val: {:.4f}'
                    .format(epoch+1, loss.item(), test_loss.item(), train_correct / train_total, correct / total, correct_val / total_val))

# # # Save the model checkpoint
#     if epoch % 20 == 0:
#       torch.save(model.state_dict(), "/content/gdrive/MyDrive/ckpt/model_epoch_{}_anchor_{}_k_{}_lr_{}.pt".format(epoch, num_anchors, k, lr))

!ls /content/gdrive/MyDrive/'Colab Notebooks'/

# Commented out IPython magic to ensure Python compatibility.
# %pycat code.py
