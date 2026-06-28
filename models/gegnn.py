# -*- coding: utf-8 -*-

import torch
from torch.nn import Linear as Lin
from torch.nn import Sigmoid
from torch.nn import Sequential as Seq
from torch_geometric.nn.conv import MessagePassing

from models.common import MLP, MLP2


class MAXGE_max_3(MessagePassing):
    def __init__(self, **kwargs):
        super(MAXGE_max_3, self).__init__(aggr="max", **kwargs)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        agg = x_j
        return agg

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        return tmp


class MAXGE_mean(MessagePassing):
    def __init__(self, **kwargs):
        super(MAXGE_mean, self).__init__(aggr="mean", **kwargs)
        self.mlp1 = MLP([17, 24, 12])

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg


class MAXGE_max_1(MessagePassing):
    def __init__(self, **kwargs):
        super(MAXGE_max_1, self).__init__(aggr="max", **kwargs)
        self.mlp1 = MLP([17, 24, 12])

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg


class MAXGE_max_2(MessagePassing):
    def __init__(self, **kwargs):
        super(MAXGE_max_2, self).__init__(aggr="max", **kwargs)
        self.mlp1 = MLP([25, 36, 18])

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg


class GEGNN(torch.nn.Module):
    def __init__(self, link_count, device):
        super(GEGNN, self).__init__()
        self.link_count = link_count
        self.device = device
        self.mlpGNN0 = MLP([3, 8])
        self.mlpGNN1 = MLP([30, 15])
        self.mlpGNN2 = MLP([30, 10])
        self.MAXGE_mean = MAXGE_mean()
        self.MAXGE_max_1 = MAXGE_max_1()
        self.MAXGE_max_2 = MAXGE_max_2()
        self.MAXGE_max_3 = MAXGE_max_3()
        self.mlpGNN5 = MLP2([48, 36, 16])
        self.mlpGNN5 = Seq(*[self.mlpGNN5, Seq(Lin(16, 1), Sigmoid())])

    def forward(self, data):
        x0, edge_attr, edge_index, y0 = data.x, data.edge_attr, data.edge_index, data.y
        x1 = self.mlpGNN0(x0[:, :3])
        edge_attr_0 = edge_attr[:, :1]

        result = self._bucket_global_features(x0)
        x2 = self.MAXGE_mean(x=x1, edge_index=edge_index, edge_attr=edge_attr_0)
        x3 = self.MAXGE_max_1(x=x1, edge_index=edge_index, edge_attr=edge_attr_0)
        x4 = self.MAXGE_max_2(x=x3, edge_index=edge_index, edge_attr=edge_attr_0)

        z0 = self.mlpGNN1(result)
        z1 = self.MAXGE_max_3(x=z0, edge_index=edge_index, edge_attr=edge_attr_0)
        z1 = self.mlpGNN2(z1)
        x5 = torch.cat((x1, x2, x4, z1), dim=1)
        out = self.mlpGNN5(x5)
        return out

    def _bucket_global_features(self, x0):
        kk = self.link_count
        w0 = x0[:, 3:]
        batnum = x0.shape[0]
        kk2 = kk + kk
        kk3 = kk2 + kk
        kk4 = kk3 + kk
        new_1, new_2, new_3, new_4 = (
            w0[:, :kk],
            w0[:, kk:kk2],
            w0[:, kk2:kk3],
            w0[:, kk3:kk4],
        )
        bucket_size = 30
        num_buckets = 300 // bucket_size
        bucket_indices = (new_1 // bucket_size).long()

        buckets_new_2 = torch.zeros((batnum, num_buckets), device=self.device)
        buckets_new_3 = torch.zeros((batnum, num_buckets), device=self.device)
        buckets_new_4 = torch.zeros((batnum, num_buckets), device=self.device)

        batch_indices = torch.arange(batnum, device=self.device).unsqueeze(1).expand(-1, kk)
        bucket_indices_expanded = bucket_indices[
            batch_indices, torch.arange(kk, device=self.device)
        ]

        buckets_new_2.scatter_add_(dim=1, index=bucket_indices_expanded, src=new_2)
        buckets_new_3.scatter_add_(dim=1, index=bucket_indices_expanded, src=new_3)
        buckets_new_4.scatter_add_(dim=1, index=bucket_indices_expanded, src=new_4)
        return torch.cat([buckets_new_2, buckets_new_3, buckets_new_4], dim=1)


class GEGNNNoGE(torch.nn.Module):
    def __init__(self):
        super(GEGNNNoGE, self).__init__()
        self.mlpGNN0 = MLP([3, 8])
        self.MAXGE_mean = MAXGE_mean()
        self.MAXGE_max_1 = MAXGE_max_1()
        self.MAXGE_max_2 = MAXGE_max_2()
        self.mlpGNN5 = MLP2([38, 36, 16])
        self.mlpGNN5 = Seq(*[self.mlpGNN5, Seq(Lin(16, 1), Sigmoid())])

    def forward(self, data):
        x0, edge_attr, edge_index, y0 = data.x, data.edge_attr, data.edge_index, data.y
        x1 = self.mlpGNN0(x0[:, :3])
        edge_attr = edge_attr[:, :1]

        x2 = self.MAXGE_mean(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        x3 = self.MAXGE_max_1(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        x4 = self.MAXGE_max_2(x=x3, edge_index=edge_index, edge_attr=edge_attr)

        x5 = torch.cat((x1, x2, x4), dim=1)
        out = self.mlpGNN5(x5)
        return out

