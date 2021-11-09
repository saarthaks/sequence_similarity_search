import torch
import h5py
import numpy as np

class SubspaceDataset:
    def __init__(self, d, D):
        self.d = d
        self.D = D

        self.means = 0.5 + 0.09*np.random.randn(d)
        self.Cov = 0.01*np.eye(d)
        self.L = (np.random.randn(d, D) + 1)/d

        self.database = None

    def generate_db(self, num_samples):
        assert(self.database is None)

        data_low = np.random.multivariate_normal(self.means, self.Cov,
                                                 size=num_samples)
        self.database_low = torch.from_numpy(data_low).float()
        data = data_low @ self.L
        self.database = torch.from_numpy(data).float()

        return self.database

    def generate_queries(self, num_queries):
        queries_low = np.random.multivariate_normal(self.means, self.Cov,
                                                    size=num_queries)
        queries = queries_low @ self.L

        return torch.from_numpy(queries).float(), torch.from_numpy(queries_low).float()

class SIFTDataset:
    def __init__(self, filepath, ntot):
        self.file = h5py.File(filepath, 'r')
        self.db_idx = None
        self.ntot = ntot
        self.db_means = None
        self.db_stds = None

    def generate_db(self):
        data = np.array(self.file.get('fea')).astype(np.float32)
        if self.db_idx is None:
            self.db_idx = np.random.choice(data.shape[0], size=self.ntot)
        samples = data[self.db_idx, :]
        self.db_means = samples.mean(axis=0)
        self.db_stds = samples.std(axis=0)
        # samples = data[:self.ntot, :]
        samples -= samples.mean(axis=0)

        return torch.from_numpy(samples).float()

    def generate_queries(self, num_queries, with_idx=False):
        data = np.array(self.file.get('fea')).astype(np.float32)
        if self.db_idx is None:
            self.db_idx = np.random.choice(data.shape[0], size=self.ntot)
        qidx = np.random.choice(self.db_idx, size=num_queries)
        queries = data[qidx, :].astype(np.float32)
        queries -= queries.mean(axis=0)
        if with_idx:
            return torch.from_numpy(queries).float(), qidx
        return torch.from_numpy(queries).float()

class BERTDataset:
    def __init__(self, filepath_q, filepath_db, ntot):
        self.file_q = filepath_q
        self.file_db = filepath_db
        self.db_idx = None
        self.ntot = ntot

    def generate_db(self):
        data = torch.load(self.file_db)
        if self.db_idx is None:
            self.db_idx = np.random.choice(data.shape[0], size=self.ntot)
        samples = data[self.db_idx, :]
        # samples = data[:self.ntot, :]
        samples -= samples.mean(dim=0)

        return samples.float()

    def generate_queries(self, num_queries, with_idx=False):
        data = torch.load(self.file_q)
        doc, embed, dim = data.shape
        data = data.reshape(doc*embed, dim)
        qidx = np.random.choice(doc*embed, size=num_queries)
        queries = data[qidx, :]
        queries -= queries.mean(dim=0)
        if with_idx:
            return queries.float(), qidx
        return queries.float()

class GLOVEDataset:
    def __init__(self, filepath, dim, ntot):
        self.file = filepath
        self.db_idx = None
        self.dim = dim
        self.ntot = ntot

    def generate_db(self):
        data = torch.load(self.file)
        if self.db_idx is None:
            self.db_idx = np.random.choice(data.shape[0], size=self.ntot)
        samples = data[self.db_idx, :]
        samples -= samples.mean(dim=0)

        return samples

    def generate_queries(self, num_queries, with_idx=False):
        data = torch.load(self.file)
        qidx = np.random.choice(data.shape[0], size=num_queries)
        queries = data[qidx, :]
        queries -= queries.mean(dim=0)
        if with_idx:
            return queries, qidx
        return queries

