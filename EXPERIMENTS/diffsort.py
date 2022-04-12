#!/usr/bin/env python
# coding: utf-8

# In[1]:

import logging
import matplotlib.pyplot as plt
import argparse

import torch, itertools
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


import sys
# sys.path.insert(1, '/Users/derekhuang/Documents/Research/sequence_similarity_search/classes')
sys.path.insert(1, '/home/users/huangda/sequence_similarity_search/classes')
import utils
from data_classes import BERTDataset
from dist_perm import DistPerm
import torchsort

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def soft_rank(array, strength=.000001):
  #  return pytorch_ops.soft_rank(array.cpu(), direction="DESCENDING", regularization_strength=.001).cuda()
   return torchsort.soft_rank(-1 * array, regularization_strength=strength)

class AnchorNet(nn.Module):
    def __init__(self, num_anchs, d, k, out=128, method="plane", top='top1', r=10):
        super(AnchorNet, self).__init__()
        # self.transform = nn.Sequential(
        #   # nn.Linear(d, hidden),
        #   # nn.ReLU(),
        #   # nn.Linear(hidden, hidden2),
        #   # nn.ReLU(),
        #   # nn.Linear(hidden, hidden2),
        #   # nn.Linear(hidden2, out)
        # )
        self.anchors = nn.Linear(d, num_anchs)
        self.k = k
        self.r = r
        self.top = top
        self.method = method

    def forward(self, data, query):
        # data_out = self.transform(data)
        # query_out = self.transform(query)
        # data_rank = torch.clamp(soft_rank(self.anchors(data_out)), max=k)
        # query_rank = torch.clamp(soft_rank(self.anchors(query_out)), max=k)
        # data_rank = soft_rank(self.anchors(data))
        # query_rank = soft_rank(self.anchors(query))
        if self.method=='plane':
            data_rank = torch.abs(self.anchors(data))
            query_rank = torch.abs(self.anchors(query))
            anchor_norm = torch.norm(self.anchors.weight, dim=1)
            data_rank = soft_rank(torch.div(data_rank, anchor_norm))
            query_rank = soft_rank(torch.div(query_rank, anchor_norm))
        elif self.method=='anchor':
            data_rank = soft_rank(self.anchors(data))
            query_rank = soft_rank(self.anchors(query))

        if self.top=='top1':
            out = torch.matmul(query_rank, data_rank.T)
        if self.top=='topk':
            # Issues with super large or super small... 
            out = torch.matmul(query_rank, data_rank.T)
            out = F.normalize(out, p=2, dim=1)
            out = torch.div(out, out.max(dim=1)[0][:,None])

            out = torch.clamp(soft_rank(out, strength=.0000001), max=self.r+1)

        return out

    def evaluate(self, data, query):
        if self.method=='plane':
            data_rank = torch.abs(self.anchors(data))
            query_rank = torch.abs(self.anchors(query))
            anchor_norm = torch.norm(self.anchors.weight, dim=1)
            d_dist = torch.div(data_rank, anchor_norm)
            q_dist = torch.div(query_rank, anchor_norm)
        elif self.method=='anchor':
            q_dist = self.anchors(query)
            d_dist = self.anchors(data)

        query_ranks = self.k*torch.ones(q_dist.shape, dtype=torch.float, device='cuda')
        data_ranks = self.k*torch.ones(d_dist.shape, dtype=torch.float, device='cuda')

        query_ids = torch.argsort(q_dist, dim=1)[:, :self.k]
        data_ids = torch.argsort(d_dist, dim=1)[:, :self.k]

        q_ids = torch.arange(query.shape[0], device='cuda')[:,None]
        d_ids = torch.arange(data.shape[0], device='cuda')[:,None]
        query_ranks[q_ids, query_ids] = torch.arange(self.k, dtype=torch.float, device='cuda')
        data_ranks[d_ids, data_ids] = torch.arange(self.k, dtype=torch.float, device='cuda')

        db_dists = torch.cdist(data_ranks, query_ranks, p=1).float()
        r_for_topr = self.r if self.top=='topk' else 10
        closest_idx = torch.topk(db_dists, r_for_topr, dim=0, largest=False)
        return closest_idx[1].transpose(0,1), query_ranks, closest_idx[0]

def dataset_split(dataset, train_frac):
    length = len(dataset)
   
    train_length = int(length * train_frac)
    valid_length = int((length - train_length) / 2)
    test_length  = length - train_length - valid_length

    sets = random_split(dataset, (train_length, valid_length, test_length))
    dataset = {name: set for name, set in zip(('train', 'val', 'test'), sets)}
    return dataset

# Fits fine in mem
def data_loaders(quers, db):
    batch_size=3200
    train_split = .8

    query_datasets = dataset_split(quers, train_split)
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

    return (query_data_loader, query_data_test_loader, query_data_val_loader, docs_loader, docs_test_loader, docs_val_loader)

def return_loader(query_data_loader, query_data_test_loader, query_data_val_loader, d, r, ret):
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
    num_neighbors = 1
    if args.top == 'topk':
        num_neighbors = r
    true = torch.tensor(index_l2.kneighbors(q, n_neighbors=num_neighbors)[1])
    query_data = []
    for i in range(q.shape[0]):
        if args.top=='topk':
            ground_truth = np.empty(len(d))
            ground_truth.fill(r+1)
            top = np.arange(r) + 1
            ground_truth.put(true[i].view(-1, 1), top)
            query_data.append([q[i], ground_truth])
        elif args.top=='top1':
            query_data.append([q[i], true[i]])
    test_set = dataset_split(query_data, 1)
#   Return data as a dataloader
    data = torch.utils.data.DataLoader(dataset=test_set['train'], 
                                           batch_size=320, 
                                           shuffle=False)
    return data

# def test_inference(docs_test_loader, top, r, model):
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for d in docs_test_loader:
#         # for d in docs_loader:
#             test_loader = return_loader(query_data_loader, query_data_test_loader, query_data_val_loader, d, r, ret='test')
#             d = d.to(device)
#             for q, l in test_loader:
#                 q = q.to(device)
#                 l = l.to(device)
#                 if top=='top1':
#                     outputs = model.evaluate(d,q)
#         #           Recall: TP / (TP + TN)
#                     predicted = outputs[0][:,0]
#                     total += l.size(0)
#                     correct += (predicted.to(device) == l.flatten()).sum().item()
#                 elif top=='topk':
#                     l = l.int()
#                     l = l - (r + 1)
#                     l = torch.nonzero(l, as_tuple=True)[1].reshape(-1, r)
#                     outputs = model.evaluate(d,q)[0]
#                     c = torch.hstack((outputs, l)).squeeze()
#                     c = c.sort(dim=1)[0]
#                     intersection = torch.sum(c[:, 1:] == c[:, :-1], dim=1) / r
#                     correct += intersection.mean()
#                     total += 1

#     logger.info ('recall_test: {:.4f}'
#             .format(correct / total))

# query_data_loader query_data_test_loader query_data_val_loader docs_loader docs_test_loader docs_val_loader
def train(model, criterion, optimizer, top, logger, r, query_data_loader, query_data_test_loader, query_data_val_loader, docs_loader, docs_test_loader, docs_val_loader):

    loss_steps = []
    test_steps = []
    for epoch in range(200):
    #   Train step: get batch of train documents, then generate the dataloader containing 
    #   the train queries and the correct data labels
        train_correct = 0
        train_total = 0
        for step, d in enumerate(docs_loader):
            query_loader = return_loader(query_data_loader, query_data_test_loader, query_data_val_loader, d, r, ret='query')
            d = d.to(device)
            for i, (q, l) in enumerate(query_loader):  
                q = q.to(device)
                l = l.to(device)
                outputs = model(d,q)
                loss = criterion(outputs, l.squeeze().float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if top=='top1':
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += l.size(0)
                    train_correct += (predicted == l.flatten()).sum().item()
                elif top=='topk':
                    l = l.int()
                    l = l - (r + 1)
                    l = torch.nonzero(l, as_tuple=True)[1].reshape(-1, r)
                    outputs = model.evaluate(d,q)[0]
                    c = torch.hstack((outputs, l)).squeeze()
                    c = c.sort(dim=1)[0]
                    intersection = torch.sum(c[:, 1:] == c[:, :-1], dim=1) / r
                    train_correct += intersection.mean()
                    train_total += 1
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
                        test_loader = return_loader(query_data_loader, query_data_test_loader, query_data_val_loader, d, r, ret='test')
                        d = d.to(device)
                        for q, l in test_loader:
                            q = q.to(device)
                            l = l.to(device)
                            lol = model(d,q)
                            test_loss = criterion(lol, l.squeeze())
                            if top=='top1':
                                outputs = model.evaluate(d,q)
                    #           Recall: TP / (TP + TN)
                                predicted = outputs[0][:,0]
                                total += l.size(0)
                                correct += (predicted.to(device) == l.flatten()).sum().item()
                            elif top=='topk':
                                l = l.int()
                                l = l - (r + 1)
                                l = torch.nonzero(l, as_tuple=True)[1].reshape(-1, r)
                                outputs = model.evaluate(d,q)[0].to(device)

                                c = torch.hstack((outputs, l)).squeeze()
                                c = c.sort(dim=1)[0]
                                intersection = torch.sum(c[:, 1:] == c[:, :-1], dim=1) / r
                                correct += intersection.mean()
                                total += 1

                    for d in docs_val_loader:
                    # for d in docs_loader:
                        test_loader = return_loader(query_data_loader, query_data_test_loader, query_data_val_loader, d, r, ret='val')
                        d = d.to(device)
                        for q, l in test_loader:
                            q = q.to(device)
                            l = l.to(device)
                            if args.top=='top1':
                                outputs = model.evaluate(d,q)
                    #           Recall: TP / (TP + TN)
                                predicted = outputs[0][:,0]
                                total_val += l.size(0)
                                correct_val += (predicted.to(device) == l.flatten()).sum().item()
                            elif args.top=='topk':
                                l = l.int()
                                l = l - (r + 1)
                                l = torch.nonzero(l, as_tuple=True)[1].reshape(-1, r)
                                outputs = model.evaluate(d,q)[0].to(device)

                                c = torch.hstack((outputs, l)).squeeze()
                                c = c.sort(dim=1)[0]
                                intersection = torch.sum(c[:, 1:] == c[:, :-1], dim=1) / r
                                correct_val += intersection.mean()
                                total_val += 1
                    
                    loss_steps.append(loss.item())
                    test_steps.append(test_loss.item())
                    logger.info ('Epoch [{}], Loss: {:.4f}, Test Loss: {:.4f}, recall_train: {:.4f}, recall_test: {:.4f} recall_val: {:.4f}'
                        .format(epoch+1, loss.item(), test_loss.item(), train_correct / train_total, correct / total, correct_val / total_val))

    return loss_steps, test_steps

def plot(loss_steps, test_steps, folder_name, session_name):
    plt.figure()
    plt.plot(loss_steps)
    plt.title('Training loss')
    plt.savefig('{}/{}_train_loss.png'.format(folder_name, session_name))
    plt.figure()
    plt.plot(test_steps)
    plt.title('Test loss')
    plt.savefig('{}/{}_test_loss.png'.format(folder_name, session_name))

def setup(num_queries, n):
    file_q = './Q.pt'
    file_db = './D.pt'
    data_source = BERTDataset(file_q, file_db, n)
    db = data_source.generate_db()
    data = np.array(db).astype(np.float32)
    queries = data_source.generate_queries(num_queries)
    quers = np.array(queries).astype(np.float32)

    return data, quers, db

def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    # log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")

def get_parser():
    parser = argparse.ArgumentParser(description="OK")
    parser.add_argument("--anchor", dest='anchor', help="number of anchors/hyperplanes", type=int, default=128)
    parser.add_argument("--n", dest='n', help="number of data points", type=int, default=500000)
    parser.add_argument("--D", dest="D", help="dimension of data", type=int, default=128)
    parser.add_argument("--queries", dest='num_queries', help="number of queries", type=int, default=3200)
    parser.add_argument("--k", dest='k', help="how many to keep", type=int, default=32)
    parser.add_argument("--method", dest='method', help="plane or anchor", type=str, default='plane')
    parser.add_argument("--top", dest='top', help="top1 or topk", type=str, default='top1')
    parser.add_argument("--r", dest='r', help="r of topr", type=int, default=10)
    parser.add_argument("--folder", dest='folder', help='folder name', type=str, default='diffsort')
    parser.add_argument("--lr", dest='lr', help='learning rate', type=float, default=.0001)


    return parser
    

def main(session_name, anchor, n, D, num_queries, k, method, top, r, folder, lr):
    data, quers, db = setup(num_queries, n)
    loaders = data_loaders(quers, db)

    model = AnchorNet(anchor, D, k, method=method, top=top, r=r).to(device)
    criterion = nn.MSELoss(reduction='sum')


    if args.top=='top1':
        criterion = nn.CrossEntropyLoss()
    elif args.top=='topk':
        criterion == nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)  
    loss_steps, test_steps = train(model, criterion, optimizer, top, logger, r, *loaders)
    plot(loss_steps, test_steps, folder, session_name)

if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    session_name = "anchor_{}_k_{}_lr_{}_method_{}_top_{}_r_{}".format(args.anchor, args.k, args.lr, args.method, args.top, args.r)

    logger = logging.getLogger()
    setup_logger(logger)
    logging.basicConfig(filename='{}/{}.log'.format(args.folder, session_name), level=logging.DEBUG)

    params = (args.anchor, args.n, args.D, args.num_queries, args.k, args.method, args.top, args.r, args.folder, args.lr)

    logger.info("anchors: {} n: {} D: {} queries: {}  k: {} method: {} top: {} r: {} folder:{} lr: {}" \
                    .format(*params))

    main(session_name, *params)
