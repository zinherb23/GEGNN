# -*- coding: utf-8 -*-

from torch.nn import BatchNorm1d as BN
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq


def MLP(channels, batch_norm=True):
    return Seq(
        *[
            Seq(Lin(channels[i - 1], channels[i]), ReLU())
            for i in range(1, len(channels))
        ]
    )


def MLP2(channels, batch_norm=True):
    return Seq(
        *[
            Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
            for i in range(1, len(channels))
        ]
    )

