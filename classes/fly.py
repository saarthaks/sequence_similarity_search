import numpy as np
import torch
class Fly:
    def __init__(self,k,m):
        """
        data: Nxd numpy array
        k: scalar
        m: dimensionality of projection space
        Note that in Flylsh, the hash length and embedding_size are NOT the same
        whereas in usual LSH they are
        """
        self.k=k
        self.m=m
        self.weights = None
            
    def fit(self, data, sampling_ratio):
        """
        data: Nxd numpy array
        sampling_ratio: fraction of input dims to sample from when producing a hash
        """
        num_projections=int(sampling_ratio*data.shape[1])
        weights=np.random.random((data.shape[1],self.m))
        yindices=np.arange(weights.shape[1])[None,:]
        xindices=weights.argsort(axis=0)[-num_projections:,:]
        self.weights=np.zeros_like(weights,dtype=np.bool)
        self.weights[xindices,yindices]= True#sparse projection vectors
        
        return self.weights
    
    def add(self, data):
        """
        data: Nxd numpy array
        """
        all_activations=(data@self.weights)
        xindices=np.arange(data.shape[0])[:,None]
        yindices=all_activations.argsort(axis=1)[:,-self.k:]
        self.hashes=np.zeros_like(all_activations,dtype=np.bool)
        self.hashes[xindices,yindices]=True #choose topk activations
        
        return self.hashes

    def search(self, queries, num, hash):
        """
        queries: Qxd numpy array
        num: number of nearest neighbors to return per d-dimensional query
        """
        self.hashes = hash
        all_activations=(queries@self.weights)
        xindices=np.arange(queries.shape[0])[:,None]
        yindices=all_activations.argsort(axis=1)[:,-self.k:]
        qhashes=np.zeros_like(all_activations,dtype=np.bool)
        qhashes[xindices,yindices]=True #choose topk activations
        qhashes[0] = qhashes[0][::-1]
        
        # L1_distances = np.array([np.sum(np.abs(qh^self.hashes),axis=1) for qh in qhashes])
        # NNs=L1_distances.argsort(axis=1)[:,:num]
        # print('self hashes' + str(self.hashes.shape))
        # print('qhashes' + str(qhashes.shape))
        dists = torch.cdist(torch.from_numpy(self.hashes.astype(float)), torch.from_numpy(qhashes.astype(float)), p=1).float()
        # print(np.any(self.hashes==0))
        NNs = torch.topk(dists, num, dim=0, largest=False)
        # print(NNs)
        return NNs[1].transpose(0, 1), qhashes, NNs[0]