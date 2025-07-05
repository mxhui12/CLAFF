import torch
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr, WebKB, Actor
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data,Batch
from torch_geometric.utils import negative_sampling
import os
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
def random_split_classes(data, train_num):
    labels = data.y.to('cpu')
    all_classes = torch.unique(labels)
    shuffled_classes = torch.randperm(len(all_classes))
    train_class = all_classes[shuffled_classes[:train_num]]
    test_class = all_classes[shuffled_classes[train_num:]]
    return train_class, test_class
def node_sample_and_save(data, k, n, graphs_list,train_class, device):
    labels = data.y.to('cpu')
    classess = torch.randperm(len(torch.unique(train_class)))[:n]
    train_idx = torch.cat([torch.where(labels == cls)[0][:k] for cls in classess])
    train_query_idx = torch.cat([torch.where(labels == cls)[0][k:] for cls in classess])
    train_graphs = [graph for graph in graphs_list if graph.index in train_idx]
    train_query_graphs = [graph for graph in graphs_list if graph.index in train_query_idx]
    train_lbls = labels[train_idx]
    train_query_lbls = labels[train_query_idx]
    return train_idx.to(device), train_query_idx.to(device),train_lbls.to(device),train_query_lbls.to(device), train_graphs,train_query_graphs,classess
def test_node_sample_and_save(data, k, n, graphs_list, test_class,device):
    labels = data.y.to('cpu')
    classes = torch.randperm(len(torch.unique(test_class)))[:n]
    support_idx = torch.cat([torch.where(labels == cls)[0][:k] for cls in classes])
    query_idx = torch.cat([torch.where(labels == cls)[0][k:] for cls in classes])
    support_graphs = [graph for graph in graphs_list if graph.index in support_idx]
    query_graphs = [graph for graph in graphs_list if graph.index in query_idx]
    support_lbls = labels[support_idx]
    query_lbls = labels[query_idx]
    return support_idx.to(device), support_lbls.to(device), query_idx.to(device), query_lbls.to(
        device), support_graphs, query_graphs,classes


def load_data(dataname):
    print(dataname)
    if dataname in ['Cora','CiteSeer']:
        dataset = Planetoid(root='data/Planetoid', name=dataname, transform=NormalizeFeatures())
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
    else:
        print("Please check dataset name.")
        exit(0)
    return data, input_dim, out_dim






