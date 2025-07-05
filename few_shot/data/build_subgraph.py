
from torch_geometric.utils import subgraph, k_hop_subgraph
import torch
import numpy as np
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.data import Data, Batch
import random
import os
from random import shuffle
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch_geometric.data import Data
import numpy as np
import pickle
from cytoolz import curry
import multiprocessing as mp
from scipy import sparse as sp
from sklearn.preprocessing import normalize, StandardScaler
from torch_geometric.data import Data, Batch

def split_subgraphs(data, dir_path, device, ppr_path='./Experiment/ppr/',
                    subgraph_size=50, n_order=10):
    from copy import deepcopy
    subgraph_generator = Subgraph(data.x, data.edge_index, ppr_path, maxsize=subgraph_size, n_order=n_order)
    subgraph_generator.build()
    induced_graph_list = []
    saved_graph_list = []
    for index in range(data.x.size(0)):
        current_label = data.y[index].item()
        nodes = subgraph_generator.neighbor[index][:subgraph_size]
        nodes = torch.unique(torch.cat([torch.LongTensor([index]).to(device), torch.tensor(nodes).to(device)]))
        sub_edge_index, _ = subgraph(nodes, data.edge_index, relabel_nodes=True)
        sub_edge_index = sub_edge_index.to(device)
        x = data.x[nodes]
        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label, index=index)
        saved_graph_list.append(deepcopy(induced_graph).to('cpu'))
        induced_graph_list.append(induced_graph)
        if index % 500 == 0:
            print(f"Processed {index} nodes.")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, f'subgraph.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(saved_graph_list, f)
        print(f"Subgraph data has been saved to {file_path}.")
class PPR:
    def __init__(self, adj_mat, maxsize=200, n_order=2, alpha=0.85):
        self.n_order = n_order
        self.maxsize = maxsize
        self.adj_mat = adj_mat
        self.P = normalize(adj_mat, norm='l1', axis=0)
        self.d = np.array(adj_mat.sum(1)).squeeze()
    def search(self, seed, alpha=0.85):
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1])
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)
        idx = scores.argsort()[::-1][:self.maxsize]
        neighbor = np.array(x.indices[idx])
        seed_idx = np.where(neighbor == seed)[0]
        if seed_idx.size == 0:
            neighbor = np.append(np.array([seed]), neighbor)
        else:
            seed_idx = seed_idx[0]
            neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]
        assert np.where(neighbor == seed)[0].size == 1
        assert np.where(neighbor == seed)[0][0] == 0
        return neighbor
    @curry
    def process(self, path, seed):
        ppr_path = os.path.join(path, 'ppr{}'.format(seed))
        if not os.path.isfile(ppr_path) or os.stat(ppr_path).st_size == 0:
            print('Processing node {}.'.format(seed))
            neighbor = self.search(seed)
            torch.save(neighbor, ppr_path)
        else:
            print('File of node {} exists.'.format(seed))
    def search_all(self, node_num, path):
        neighbor = {}
        if os.path.isfile(path + '_neighbor') and os.stat(path + '_neighbor').st_size != 0:
            print("Exists neighbor file")
            neighbor = torch.load(path + '_neighbor')
        else:
            print("Extracting subgraphs")
            os.system('mkdir {}'.format(path))
            with mp.Pool() as pool:
                list(pool.imap_unordered(self.process(path), list(range(node_num)), chunksize=1000))
            print("Finish Extracting")
            for i in range(node_num):
                neighbor[i] = torch.load(os.path.join(path, 'ppr{}'.format(i)))
            torch.save(neighbor, path + '_neighbor')
            os.system('rm -r {}'.format(path))
            print("Finish Writing")
        return neighbor
class Subgraph:
    def __init__(self, x, edge_index, path, maxsize=50, n_order=10):
        self.x = x
        self.path = path
        self.edge_index = np.array(edge_index.cpu())
        self.edge_num = edge_index[0].size(0)
        self.node_num = x.size(0)
        self.maxsize = maxsize
        self.sp_adj = sp.csc_matrix((np.ones(self.edge_num), (edge_index[0].cpu(), edge_index[1].cpu())),
                                    shape=[self.node_num, self.node_num])
        self.ppr = PPR(self.sp_adj, n_order=n_order)
        self.neighbor = {}
        self.adj_list = {}
        self.subgraph = {}
    def process_adj_list(self):
        for i in range(self.node_num):
            self.adj_list[i] = set()
        for i in range(self.edge_num):
            u, v = self.edge_index[0][i], self.edge_index[1][i]
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)
    def adjust_edge(self, idx):
        dic = {}
        for i in range(len(idx)):
            dic[idx[i]] = i
        new_index = [[], []]
        nodes = set(idx)
        for i in idx:
            edge = list(self.adj_list[i] & nodes)
            edge = [dic[_] for _ in edge]
            new_index[0] += len(edge) * [dic[i]]
            new_index[1] += edge
        return torch.LongTensor(new_index)
    def adjust_x(self, idx):
        return self.x[idx]
    def build(self):
        self.neighbor = self.ppr.search_all(self.node_num, self.path)
        self.process_adj_list()
    def search(self, node_list):
        batch = []
        index = []
        size = 0
        for node in node_list:
            batch.append(self.subgraph[node])
            index.append(size)
            size += self.subgraph[node].x.size(0)
        index = torch.tensor(index)
        batch = Batch().from_data_list(batch)
        return batch, index