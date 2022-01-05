import torch
import numpy as np
from sklearn.cluster import KMeans

class DistPerm:
    def __init__(self, m, k=None, dist='l2'):
        """
        m: total number of anchors (int)
        k: number of anchors/signals used for measuring similarity (int <= m)
        dist: 'l2' (default) or 'dot' are different methods of computing distance between
              datapoint and anchor point
        """
        self.m = m

        # if k <= m is specified
        if k and k <= m:
            self.k = k
        # if k is unspecified, default to max
        else:
            self.k = m

        if dist == 'l2':
            self.dist_fn = lambda X, Y: torch.cdist(X, Y, p=2).float()
        elif dist == 'dot':
            self.dist_fn = lambda X, Y: -X @ torch.transpose(Y, dim0=0, dim1=1).float()
        else:
            # this should never happen
            self.dist_fn = None

        # initialize anchors
        self.anchors = m*[None]

        # initialize index
        self.rank_index = []

        # boolean to ensure anchors were chosen before storing data
        self.is_trained = False

    def fit(self, training_data, alg='random', fly=None):
        if alg == 'random':
            self.anchors = training_data[np.random.choice(len(training_data),
                                                          size=self.m,
                                                          replace=False)]
            print(self.anchors.shape)
            self.is_trained = True
            return self.anchors
        elif alg == 'fft':
            anchor_idx = self.farthest_first(training_data)
            self.anchors = training_data[anchor_idx]
            self.is_trained = True
            return self.anchors
        elif alg == 'weighted':
            kmeans = KMeans(n_clusters=self.m)
            weights = self.occurence_weights(training_data).numpy()
            kmeans.fit(training_data.numpy(), sample_weight=weights)
            self.anchors = torch.from_numpy(kmeans.cluster_centers_)
            self.is_trained = True
            return self.anchors
        elif alg == 'fly':
            self.anchors = torch.from_numpy(fly).T
            self.anchors = self.anchors.to(torch.float)
            self.is_trained = True
            return self.anchors
        else:
            return NotImplementedError

    def add(self, database):
        '''
        database: N x d matrix of N d-dimensional items (torch.Tensor)
        '''
        # require .fit() to select anchors
        assert(self.is_trained)

        # compute distances
        anchor_distances = self.dist_fn(database, self.anchors)
        # initialize rank vector index 
        # initialized to k, the number of anchors used for similarity search
        self.rank_index = self.k*torch.ones((database.shape[0], self.m), dtype=torch.float)
        # find closest k anchor ids, in order from closest to farthest
        closest_anchor_ids = torch.argsort(anchor_distances, dim=1)[:, :self.k]
        
        db_ids = torch.arange(database.shape[0])[:,None]
        # assign ranks from [0,k-1] to closest k anchors
        self.rank_index[db_ids, closest_anchor_ids] = torch.arange(self.k, dtype=torch.float)

        return self.rank_index

    def search(self, query, num):
        '''
        query: N x d matrix of N d-dimensional query vectors (torch.Tensor)
        num: number of closest points to return (int)
        '''
        # require .fit() to select anchors
        assert(self.is_trained)


        # compute distances
        q_dist = self.dist_fn(query, self.anchors)
        # initialize query rankings
        # initialized to k, the number of anchors used for similarity search
        query_ranks = self.k*torch.ones((query.shape[0], self.m), dtype=torch.float)
        # find closest k anchor ids, in order from closest to farthest
        closest_anchor_ids = torch.argsort(q_dist, dim=1)[:, :self.k]

        q_ids = torch.arange(query.shape[0])[:,None]
        # assign ranks from [0, k-1] to closest k anchors
        query_ranks[q_ids, closest_anchor_ids] = torch.arange(self.k, dtype=torch.float)

        # compute distances between rank vectors 
        # (L2 over rank vectors is equal to Spearman Rho Rank Correlation)

        # db_dists = torch.cdist(self.rank_index, query_ranks, p=2)
        # -X @ torch.transpose(Y, dim0=0, dim1=1).float()
        # print(self.rank_index.shape)
        # print(query_ranks.shape)
        # print(db_dists.shape)
        # db_dists = -self.rank_index @ torch.transpose(query_ranks, dim0=0, dim1=1).float()
        # print('rank index' + str(self.rank_index.shape))
        # print('query ranks' + str(query_ranks.shape))
        # print(db_dists.shape)

        # find indices of closest 'num' datapoints in db for each query
        # with shape (query.shape[0], num)
        db_dists = torch.cdist(self.rank_index, query_ranks, p=1).float()
        closest_idx = torch.topk(db_dists, num, dim=0, largest=False)
        # print(closest_idx.shape)
        # closest_idx = db_dists.argsort(axis=1)[:,:num]
        # print(closest_idx)

        return closest_idx[1].transpose(0,1), query_ranks, closest_idx[0]

    def farthest_first(self, X):
        n = X.shape[0]
        D = torch.cdist(X, X, p=2).float().numpy()
        i = np.int32(np.random.uniform(n))
        visited = [i]
        while len(visited) < self.m:
            dist = np.mean([D[i] for i in visited], 0)
            for i in np.argsort(dist)[::-1]:
                if i not in visited:
                    visited.append(i)
                    break

        return np.array(visited)
    
    def occurence_weights(self, X):
        n = X.shape[0]
        D = torch.cdist(X, X, p=2).float()
        counts = n - torch.count_nonzero(D, dim=0) + 1
        return 1/counts
