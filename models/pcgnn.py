# -*- coding: utf-8 -*-

import torch
from torch.nn import Linear as Lin
from torch.nn import Sigmoid
from torch.nn import Sequential as Seq
from torch_geometric.nn.conv import MessagePassing

from models.common import MLP


class PCGNNConv1(MessagePassing):
    def __init__(self, **kwargs):
        super(PCGNNConv1, self).__init__(aggr="max", **kwargs)
        self.mlp1 = MLP([24, 24, 16])

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.mlp1, self.mlp2)


class PCGNNConv2(MessagePassing):
    def __init__(self, **kwargs):
        super(PCGNNConv2, self).__init__(aggr="max", **kwargs)
        self.mlp1 = MLP([40, 32, 24])

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.mlp1, self.mlp2)


class PCGNN(torch.nn.Module):
    def __init__(self):
        super(PCGNN, self).__init__()
        self.mlp1 = MLP([3, 8])
        self.mlp3 = MLP([2, 8])
        self.mlp2 = MLP([32, 32, 16])
        self.mlp2 = Seq(*[self.mlp2, Seq(Lin(16, 1), Sigmoid())])
        self.PCGNNConv1 = PCGNNConv1()
        self.PCGNNConv2 = PCGNNConv2()

    def forward(self, data):
        x0, edge_attr, edge_index, y0 = data.x, data.edge_attr, data.edge_index, data.y
        x0 = self.mlp1(x0[:, :3])
        edge_attr = self.mlp3(edge_attr[:, 1:3])
        x1 = self.PCGNNConv1(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        x2 = self.PCGNNConv2(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        x3 = torch.cat([x0, x2], axis=1)
        out = self.mlp2(x3)
        return out

