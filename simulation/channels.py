# -*- coding: utf-8 -*-

import numpy as np


def compute_path_losses(general_para, distances):
    N = np.shape(distances)[-1]
    assert N == general_para.n_links
    h1 = general_para.tx_height
    h2 = general_para.rx_height
    signal_lambda = 2.998e8 / general_para.carrier_f
    antenna_gain_decibel = general_para.antenna_gain_decibel
    Rbp = 4 * h1 * h2 / signal_lambda
    Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * h1 * h2)))
    sum_term = 20 * np.log10(distances / Rbp)
    Tx_over_Rx = Lbp + 6 + sum_term + ((distances > Rbp).astype(int)) * sum_term
    pathlosses = -Tx_over_Rx + np.eye(N) * antenna_gain_decibel
    pathlosses = np.power(10, (pathlosses / 10))
    return pathlosses


def proc_train_losses(train_path_losses, train_channel_losses, train_K):
    mask = np.eye(train_K)
    diag_path = np.multiply(mask, train_path_losses)
    off_diag_path = train_path_losses - diag_path
    diag_channel = np.multiply(mask, train_channel_losses)
    train_losses = diag_channel + off_diag_path
    return train_losses


def add_shadowing(channel_losses):
    shadow_coefficients = np.random.normal(loc=0, scale=8, size=np.shape(channel_losses))
    channel_losses = channel_losses * np.power(10.0, shadow_coefficients / 10)
    return channel_losses


def add_fast_fading(channel_losses):
    fastfadings = (
        np.power(np.random.normal(loc=0, scale=1, size=np.shape(channel_losses)), 2)
        + np.power(np.random.normal(loc=0, scale=1, size=np.shape(channel_losses)), 2)
    ) / 2
    channel_losses = channel_losses * fastfadings
    return channel_losses


def get_directLink_channel_losses(channel_losses):
    return np.diagonal(channel_losses, axis1=1, axis2=2)


def get_crossLink_channel_losses(channel_losses):
    N = np.shape(channel_losses)[-1]
    return channel_losses * ((np.identity(N) < 1).astype(float))

