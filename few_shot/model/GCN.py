import torch as th
import torch.nn.functional as F
import torch, gc
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_max_pool, GlobalAttention
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np


def act(x=None, act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        return torch.nn.LeakyReLU() if x is None else F.leaky_relu(x)
    elif act_type == 'tanh':
        return torch.nn.Tanh() if x is None else torch.tanh(x)
    elif act_type == 'relu':
        return torch.nn.ReLU() if x is None else F.relu(x)
    elif act_type == 'sigmoid':
        return torch.nn.Sigmoid() if x is None else torch.sigmoid(x)
    elif act_type == 'softmax':
        return torch.nn.Softmax(dim=-1) if x is None else F.softmax(x, dim=-1)
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, num_layer=3,JK="last", drop_ratio=0, pool='mean'):
        super().__init__()
        GraphConv = GCNConv
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)
        if out_dim is None:
            out_dim = hid_dim
        if num_layer < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(num_layer))
        elif num_layer == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        self.JK = JK
        self.drop_ratio = drop_ratio
        if pool == "sum":
            self.pool = global_add_pool
        elif pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
    def forward(self, x, edge_index, batch = None, tuing = None):
        h_list = [x]
        for idx, conv in enumerate(self.conv_layers[0:-1]):
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, self.drop_ratio, training=self.training)
            h_list.append(x)
        x = self.conv_layers[-1](x, edge_index)
        h_list.append(x)
        if self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_emb = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]
        if batch == None:
            return node_emb
        else:
            if tuing is not None:
                node_emb = tuing(node_emb)
            graph_emb = self.pool(node_emb, batch.long())
            return graph_emb
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()