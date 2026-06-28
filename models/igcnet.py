# -*- coding: utf-8 -*-

import torch
from torch.nn import Linear as Lin
from torch.nn import Sigmoid
from torch.nn import Sequential as Seq
from torch_geometric.nn.conv import MessagePassing

from models.common import MLP


class IGConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(IGConv, self).__init__(aggr="max", **kwargs)
        self.mlp1 = mlp1
        self.mlp2 = mlp2

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp)
        return torch.cat([x[:, :2], comb], dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.mlp1, self.mlp2)


class IGCNet(torch.nn.Module):
    def __init__(self):
        super(IGCNet, self).__init__()
        self.mlp1 = MLP([4, 32, 32])
        self.mlp2 = MLP([35, 35, 16])
        self.mlp2 = Seq(*[self.mlp2, Seq(Lin(16, 1), Sigmoid())])
        self.conv = IGConv(self.mlp1, self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index, y0 = data.x, data.edge_attr, data.edge_index, data.y
        x0 = x0[:, :3]
        edge_attr = edge_attr[:, :1]
        x1 = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        out = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        out = out[:, 2]
        return out

