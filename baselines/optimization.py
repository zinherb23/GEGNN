# -*- coding: utf-8 -*-

import numpy as np

from simulation.channels import (
    get_crossLink_channel_losses,
    get_directLink_channel_losses,
)


def simple_greedy(X, AAA, label, test_K):
    n = X.shape[0]
    thd = int(np.sum(label) / n)
    Y = np.ones((n, test_K))
    return Y


def FP_optimize(general_para, g, weights):
    number_of_samples, N, _ = np.shape(g)
    assert np.shape(g) == (number_of_samples, N, N)
    assert np.shape(weights) == (number_of_samples, N)
    g_diag = get_directLink_channel_losses(g)
    g_nondiag = get_crossLink_channel_losses(g)
    weights = np.expand_dims(weights, axis=-1)
    g_diag = np.expand_dims(g_diag, axis=-1)
    x = np.ones([number_of_samples, N, 1])
    tx_power = general_para.tx_power
    output_noise_power = general_para.output_noise_power
    tx_powers = np.ones([number_of_samples, N, 1]) * tx_power
    for i in range(1):
        x = _fp_iteration(
            g, g_diag, g_nondiag, weights, x, tx_powers, output_noise_power
        )
    assert np.shape(x) == (number_of_samples, N, 1)
    x_final = np.squeeze(x, axis=-1)
    return x_final


def _fp_iteration(g, g_diag, g_nondiag, weights, x, tx_powers, output_noise_power):
    p_x_prod = x * tx_powers
    z_denominator = np.matmul(g_nondiag, p_x_prod) + output_noise_power
    z_numerator = g_diag * p_x_prod
    z = z_numerator / z_denominator
    y_denominator = np.matmul(g, p_x_prod) + output_noise_power
    y_numerator = np.sqrt(z_numerator * weights * (z + 1))
    y = y_numerator / y_denominator
    x_denominator = np.matmul(np.transpose(g, (0, 2, 1)), np.power(y, 2)) * tx_powers
    x_numerator = y * np.sqrt(weights * (z + 1) * g_diag * tx_powers)
    x_new = np.power(x_numerator / x_denominator, 2)
    x_new[x_new > 1] = 1
    return x_new


def batch_WMMSE2_1(p_int, alpha, H, Pmax, var_noise):
    return _batch_WMMSE2(p_int, alpha, H, Pmax, var_noise, iterations=1)


def batch_WMMSE2(p_int, alpha, H, Pmax, var_noise):
    return _batch_WMMSE2(p_int, alpha, H, Pmax, var_noise, iterations=100)


def _batch_WMMSE2(p_int, alpha, H, Pmax, var_noise, iterations):
    N = p_int.shape[0]
    K = p_int.shape[1]
    vnew = 0
    b = np.sqrt(p_int)
    f = np.zeros((N, K, 1))
    w = np.zeros((N, K, 1))

    mask = np.eye(K)
    f, w = _wmmse_receiver_weights(H, b, mask, var_noise)

    for ii in range(iterations):
        b = _wmmse_update_transmitter(alpha, H, Pmax, b, f, w, N, K)
        f, w = _wmmse_receiver_weights(H, b, mask, var_noise)
    p_opt = np.square(b)
    return p_opt


def _wmmse_receiver_weights(H, b, mask, var_noise):
    bp = b if b.ndim == 3 else np.expand_dims(b, 1)
    rx_power = np.multiply(H, bp)
    rx_power_s = np.square(rx_power)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
    interference = np.sum(rx_power_s, 2) + var_noise
    f = np.divide(valid_rx_power, interference)
    w = 1 / (1 - np.multiply(f, valid_rx_power))
    return f, w


def _wmmse_update_transmitter(alpha, H, Pmax, b, f, w, N, K):
    fp = np.expand_dims(f, 1)
    rx_power = np.multiply(H.transpose(0, 2, 1), fp)
    mask = np.eye(K)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)
    bup = np.multiply(alpha, np.multiply(w, valid_rx_power))
    rx_power_s = np.square(rx_power)
    wp = np.expand_dims(w, 1)
    alphap = np.expand_dims(alpha, 1)
    bdown = np.sum(np.multiply(alphap, np.multiply(rx_power_s, wp)), 2)
    btmp = bup / bdown
    return (
        np.minimum(btmp, np.ones((N, K)) * np.sqrt(Pmax))
        + np.maximum(btmp, np.zeros((N, K)))
        - btmp
    )

