# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch_geometric.data import Data


def build_graph(loss, dist, norm_dist, norm_loss, K, threshold):
    edge_index, edge_attr = _build_edges(dist, norm_dist, norm_loss, K, threshold)
    y = torch.tensor(np.expand_dims(loss, axis=0), dtype=torch.float)
    x = _build_node_features(dist, norm_dist, norm_loss, K, threshold)
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=y)
    return data


def proc_data(HH, dists, norm_dists, norm_HH, K, threshold=300):
    n = HH.shape[0]
    data_list = []
    for i in range(n):
        data = build_graph(
            HH[i, :, :],
            dists[i, :, :],
            norm_dists[i, :, :],
            norm_HH[i, :, :],
            K,
            threshold,
        )
        data_list.append(data)
    return data_list


def _build_edges(dist, norm_dist, norm_loss, K, threshold):
    dist2 = _threshold_off_diagonal_distances(dist, K, threshold)
    attr_ind = np.nonzero(dist2)
    attr_ind_1 = np.array(attr_ind).astype(int)
    attr_ind_2T = _build_transposed_attr_indices(attr_ind)

    e1 = np.expand_dims(norm_dist[tuple(attr_ind_1)], axis=1)
    e2 = np.expand_dims(norm_loss[tuple(attr_ind_1)], axis=1)
    e3 = np.expand_dims(norm_loss[tuple(attr_ind_2T)], axis=1)
    edge_attr = np.concatenate((e1, e2, e3), axis=1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    edge_index = _build_edge_index(attr_ind)
    return edge_index, edge_attr


def _build_transposed_attr_indices(attr_ind):
    attr_ind_2 = np.array(attr_ind)
    attr_ind_2T = np.zeros(attr_ind_2.shape, dtype=int)
    attr_ind_2T[0, :] = attr_ind_2[1, :]
    return attr_ind_2T.astype(int)


def _build_edge_index(attr_ind):
    attr_ind = np.array(attr_ind)
    adj = np.zeros(attr_ind.shape)
    adj[0, :] = attr_ind[1, :]
    adj[1, :] = attr_ind[0, :]
    return torch.tensor(adj, dtype=torch.long)


def _build_node_features(dist, norm_dist, norm_loss, K, threshold):
    dist_B = _threshold_off_diagonal_distances(dist, K, threshold)
    dist_C = np.copy(dist)
    dist_D = np.copy(norm_dist)
    loss_E = np.copy(norm_loss)
    loss_F = np.copy(norm_loss).T
    dist_C[dist_B == 0] = 0
    dist_D[dist_B == 0] = 0
    dist_D[dist_D < 0] = 0
    loss_E[dist_B == 0] = 0
    loss_E[loss_E < 0] = 0
    loss_F[dist_B == 0] = 0
    loss_F[loss_F < 0] = 0

    x1 = np.expand_dims(np.diag(norm_dist), axis=1)
    x2 = np.expand_dims(np.diag(norm_loss), axis=1)
    x3 = np.zeros((K, 1))
    x = np.concatenate((x1, x2, x3), axis=1)
    x = np.concatenate((x, dist_C, dist_D, loss_E, loss_F), axis=1)
    return torch.tensor(x, dtype=torch.float)


def _threshold_off_diagonal_distances(dist, K, threshold):
    dist2 = np.copy(dist)
    mask = np.eye(K)
    diag_dist = np.multiply(mask, dist2)
    dist2 = dist2 + 1000 * diag_dist
    dist2[dist2 > threshold] = 0
    return dist2

