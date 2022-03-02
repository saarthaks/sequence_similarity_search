#!/usr/bin/env python
# coding: utf-8

# In[1]:

import logging
import matplotlib.pyplot as plt
import argparse

import torch.nn.functional as F
import torch
import torch.nn as nn
import sys
# sys.path.insert(1, '/Users/derekhuang/Documents/Research/sequence_similarity_search/classes')
sys.path.insert(1, '/home/users/huangda/sequence_similarity_search/classes')

# sys.path.insert(1, '/Users/derekhuang/Documents/Research/fast-soft-sort/fast_soft_sort')
from data_classes import BERTDataset
from dist_perm import DistPerm
import utils
# import pytorch_ops
import torchsort
from sklearn.neighbors import NearestNeighbors
import numpy as np

def soft_rank(array):
  #  return pytorch_ops.soft_rank(array.cpu(), direction="DESCENDING", regularization_strength=.001).cuda()
   return torchsort.soft_rank(-1 * array, regularization_strength=.000001)

class AnchorNet(nn.Module):
    def __init__(self, num_anchs, d, k, out=128, method="plane"):
        super(AnchorNet, self).__init__()
        self.transform = nn.Sequential(
          # nn.Linear(d, hidden),
          # nn.ReLU(),
          # nn.Linear(hidden, hidden2),
          # nn.ReLU(),
          # nn.Linear(hidden, hidden2),
          # nn.Linear(hidden2, out)
        )
        self.anchors = nn.Linear(d, num_anchs)
        self.k = k

    def forward(self, data, query):
        # data_out = self.transform(data)
        # query_out = self.transform(query)
        # data_rank = torch.clamp(soft_rank(self.anchors(data_out)), max=k)
        # query_rank = torch.clamp(soft_rank(self.anchors(query_out)), max=k)
        # data_rank = soft_rank(self.anchors(data))
        # query_rank = soft_rank(self.anchors(query))
        if method=='plane':
            data_rank = self.anchors(data)
            query_rank = self.anchors(query)
            anchor_norm = torch.norm(self.anchors.weight, dim=0)
            data_rank = soft_rank(torch.div(data_rank, anchor_norm))
            query_rank = soft_rank(torch.div(query_rank, anchor_norm))
            out = torch.matmul(query_rank, data_rank.T)
        elif method=='anchor':
            data_rank = soft_rank(self.anchors(data))
            query_rank = soft_rank(self.anchors(query))
            out = torch.matmul(query_rank, data_rank.T)
        return out

    def evaluate(self, data, query):
        if method=='plane':
            data_rank = self.anchors(data)
            query_rank = self.anchors(query)
            anchor_norm = torch.norm(self.anchors.weight, dim=0)
            d_dist = torch.div(data_rank, anchor_norm)
            q_dist = torch.div(query_rank, anchor_norm)
        elif method=='anchor':
            q_dist = self.anchors(query)
            d_dist = self.anchors(data)

        query_ranks = self.k*torch.ones(q_dist.shape, dtype=torch.float)
        data_ranks = self.k*torch.ones(d_dist.shape, dtype=torch.float)

        query_ids = torch.argsort(q_dist, dim=1)[:, :self.k]
        data_ids = torch.argsort(d_dist, dim=1)[:, :self.k]

        q_ids = torch.arange(query.shape[0])[:,None]
        d_ids = torch.arange(data.shape[0])[:,None]
        query_ranks[q_ids, query_ids] = torch.arange(self.k, dtype=torch.float)
        data_ranks[d_ids, data_ids] = torch.arange(self.k, dtype=torch.float)

        db_dists = torch.cdist(data_ranks, query_ranks, p=1).float()
        closest_idx = torch.topk(db_dists, 10, dim=0, largest=False)
        return closest_idx[1].transpose(0,1), query_ranks, closest_idx[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")

def get_parser():
    '''get_parser returns the arg parse object, for use by an external application (and this script)
    '''
    parser = argparse.ArgumentParser(
    description="This is a description of your tool's functionality.")
    parser.add_argument("--anchor", dest='anchor', help="number of anchors/hyperplanes", type=int, default=128)
    parser.add_argument("--k", dest='k', help="how many to keep", type=int, default=32)
    parser.add_argument("--method", dest='method', help="plane or anchor", type=str, default='plane')

    return parser

parser = get_parser()
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

n = 500000
D = 128
num_queries = 3200
num_anchors = args.anchor
k = args.k
method = args.method
lr=.0001


# In[3]:


file_q = './Q.pt'
file_db = './D.pt'
data_source = BERTDataset(file_q, file_db, n)
db = data_source.generate_db()
data = np.array(db).astype(np.float32)
queries = data_source.generate_queries(num_queries)
quers = np.array(queries).astype(np.float32)

session_name = "anchor_{}_k_{}_lr_{}_method_{}.pt".format(num_anchors, k, lr, method)

logger = logging.getLogger()
setup_logger(logger)
logging.basicConfig(filename='diffsort/{}.log'.format(session_name), level=logging.DEBUG)

logger.info("n: {} D: {} queries: {} anchors: {} k: {} method: {} lr: {}".format(n, D, num_queries, num_anchors, k, method, lr))


# In[4]:


# Helper function for splitting into datasets 

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


# In[5]:


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
                                           batch_size=25000, 
                                           shuffle=False)

# The test docs
docs_test_loader = torch.utils.data.DataLoader(dataset=doc_datasets['test'], 
                                           batch_size=25000, 
                                           shuffle=False)

docs_val_loader = torch.utils.data.DataLoader(dataset=doc_datasets['val'], 
                                           batch_size=25000, 
                                           shuffle=False)


# In[6]:


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
    true = torch.tensor(index_l2.kneighbors(q, n_neighbors=1)[1])
    query_data = []
    for i in range(q.shape[0]):
        query_data.append([q[i], true[i]])
    test_set = dataset_split(query_data, 1)
#   Return data as a dataloader
    data = torch.utils.data.DataLoader(dataset=test_set['train'], 
                                           batch_size=320, 
                                           shuffle=False)
    return data


# In[7]:


model = AnchorNet(num_anchors, D, k, method=method).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)  


# In[8]:


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
            predicted = outputs[0][:,0]
            total += l.size(0)
            correct += (predicted.to(device) == l.flatten()).sum().item()

    logger.info ('recall_test: {:.4f}'
            .format(correct / total))


# In[ ]:

loss_steps = []
test_steps = []
# train
for epoch in range(200):
#   Train step: get batch of train documents, then generate the dataloader containing 
#   the train queries and the correct data labels
    train_correct = 0
    train_total = 0
    for step, d in enumerate(docs_loader):
        query_loader = return_loader(d, ret='query')
        d = d.to(device)
        for i, (q, l) in enumerate(query_loader):  
            q = q.to(device)
            l = l.to(device)
            outputs = model(d,q)
            loss = criterion(outputs, l.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += l.size(0)
            train_correct += (predicted == l.flatten()).sum().item()
        # print('Epoch [{}] Step [{}] Loss: {:.4f}'.format(epoch, step, loss.item()))
        
#   Eval step: repeat but with the test documents and test queries      
    if epoch % 1 == 0:
        with torch.no_grad():
            correct = 0
            total = 0
            correct_val = 0
            total_val = 0
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
                    total += l.size(0)
                    correct += (predicted.to(device) == l.flatten()).sum().item()


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
                    total_val += l.size(0)
                    correct_val += (predicted.to(device) == l.flatten()).sum().item()
            
            loss_steps.append(loss.item())
            test_steps.append(test_loss.item())

            logger.info ('Epoch [{}], Loss: {:.4f}, Test Loss: {:.4f}, recall_train: {:.4f}, recall_test: {:.4f} recall_val: {:.4f}'
                    .format(epoch+1, loss.item(), test_loss.item(), train_correct / train_total, correct / total, correct_val / total_val))
plt.figure()
plt.plot(loss_steps)
plt.title('Training loss')
plt.savefig('diffsort/{}_train_loss.png'.format(session_name))
plt.figure()
plt.plot(test_steps)
plt.title('Test loss')
plt.savefig('diffsort/{}_test_loss.png'.format(session_name))





