# -*- coding: utf-8 -*-

import torch


def sr_loss(data, out, K, var, device):
    power = out
    power = torch.reshape(power, (-1, K, 1))

    abs_H_2 = data.y
    abs_H_2 = abs_H_2.permute(0, 2, 1)
    rx_power = torch.mul(abs_H_2, power)
    mask = torch.eye(K)
    mask = mask.to(device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
    rate = torch.log2(1 + torch.div(valid_rx_power, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))
    loss = torch.neg(sum_rate)
    return loss

