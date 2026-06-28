# -*- coding: utf-8 -*-

import numpy as np


def normalize_data(
    train_data, test_data, train_K, test_K, train_layouts, test_layouts
):
    train_stats = _fit_train_normalization(train_data, train_K, train_layouts)
    norm_train = _apply_normalization(train_data, train_K, *train_stats)
    norm_test = _apply_normalization(test_data, test_K, *train_stats)
    return norm_train, norm_test


def _fit_train_normalization(train_data, train_K, train_layouts):
    mask = np.eye(train_K)
    train_copy = np.copy(train_data)
    diag_H = np.multiply(mask, train_copy)
    diag_mean = np.sum(diag_H) / train_layouts / train_K
    diag_var = np.sqrt(np.sum(np.square(diag_H)) / train_layouts / train_K)
    off_diag = train_copy - diag_H
    off_diag_mean = np.sum(off_diag) / train_layouts / train_K / (train_K - 1)
    off_diag_var = np.sqrt(
        np.sum(np.square(off_diag)) / train_layouts / train_K / (train_K - 1)
    )
    return diag_mean, diag_var, off_diag_mean, off_diag_var


def _apply_normalization(data, K, diag_mean, diag_var, off_diag_mean, off_diag_var):
    mask = np.eye(K)
    data_copy = np.copy(data)
    diag_H = np.multiply(mask, data_copy)
    tmp_diag = (diag_H - diag_mean) / diag_var
    off_diag = data_copy - diag_H
    tmp_off = (off_diag - off_diag_mean) / off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off, mask)
    return np.multiply(tmp_diag, mask) + tmp_off_diag

