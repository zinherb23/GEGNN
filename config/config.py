# -*- coding: utf-8 -*-

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WirelessConfig:
    n_links: int
    field_length: int = 1000
    shortest_directLink_length: int = 2
    longest_directLink_length: int = 65
    shortest_directLink_length_test: int = 2
    longest_directLink_length_test: int = 65
    shortest_crossLink_length: int = 1
    bandwidth: float = 5e6
    carrier_f: float = 2.4e9
    tx_height: float = 1.5
    rx_height: float = 1.5
    antenna_gain_decibel: float = 2.5
    tx_power_milli_decibel: int = 40
    noise_density_milli_decibel: int = -169
    normalize_noise: float = -0.13
    normalize_len300: float = 0.06
    SNR_gap_dB: int = 6
    cell_length: int = 5

    tx_power: float = field(init=False)
    input_noise_power: float = field(init=False)
    output_noise_power: float = field(init=False)
    mu: float = field(init=False)
    sigma: float = field(init=False)
    mu_pair_dist: float = field(init=False)
    sigma_pair_dist: float = field(init=False)
    SNR_gap: float = field(init=False)
    setting_str: str = field(init=False)
    n_grids: int = field(init=False)

    def __post_init__(self):
        self.tx_power = np.power(10, (self.tx_power_milli_decibel - 30) / 10)
        self.input_noise_power = (
            np.power(10, ((self.noise_density_milli_decibel - 30) / 10))
            * self.bandwidth
        )
        self.output_noise_power = self.input_noise_power
        self.mu = self.field_length / 2
        self.sigma = self.field_length / 10
        self.mu_pair_dist = (
            self.shortest_directLink_length + self.longest_directLink_length
        ) / 2
        self.sigma_pair_dist = (
            self.longest_directLink_length - self.shortest_directLink_length
        ) / 6
        self.SNR_gap = np.power(10, self.SNR_gap_dB / 10)
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(
            self.n_links,
            self.field_length,
            self.field_length,
            self.shortest_directLink_length,
            self.longest_directLink_length,
        )
        self.n_grids = np.round(self.field_length / self.cell_length).astype(int)


@dataclass(frozen=True)
class ExperimentConfig:
    train_K: int = 20
    test_K: int = 20
    train_layouts: int = 2000
    test_layouts: int = 2000
    train_batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.9
    graph_threshold: int = 300
    data_loader_workers: int = 1

