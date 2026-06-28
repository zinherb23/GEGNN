# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch_geometric.data import DataLoader

from baselines.optimization import (
    batch_WMMSE2,
    batch_WMMSE2_1,
    simple_greedy,
)
from config.config import ExperimentConfig, WirelessConfig
from graph.dataset import proc_data
from metrics.rates import compute_rates
from models.gegnn import GEGNN, GEGNNNoGE
from models.igcnet import IGCNet
from models.pcgnn import PCGNN
from simulation.channels import (
    add_fast_fading,
    add_shadowing,
    compute_path_losses,
    get_crossLink_channel_losses,
    get_directLink_channel_losses,
    proc_train_losses,
)
from simulation.layouts import generate_layouts, generate_layouts_test
from simulation.normalization import normalize_data
from training.loops import train_model


def run_experiment(config=None):
    config = config or ExperimentConfig()
    datasets = prepare_datasets(config)
    results_GNN = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _train_and_record(GEGNN(config.train_K, device), "GEGNN", datasets, config, device, results_GNN)
    _train_and_record(GEGNNNoGE(), "GEGNNnoGE", datasets, config, device, results_GNN)
    _train_and_record(PCGNN(), "PCGNN", datasets, config, device, results_GNN)

    baseline_results = run_baselines(datasets, config)
    _train_and_record(IGCNet(), "WCGCN", datasets, config, device, results_GNN)

    print_results(results_GNN, baseline_results)
    return results_GNN, baseline_results


def prepare_datasets(config):
    train_config = WirelessConfig(config.train_K)
    var = train_config.output_noise_power / train_config.tx_power
    train_coor, train_dists = generate_layouts(train_config, config.train_layouts)
    train_path_losses = compute_path_losses(train_config, train_dists)
    train_channel_losses = add_shadowing(train_path_losses)
    train_channel_losses = add_fast_fading(train_channel_losses)
    train_losses = proc_train_losses(
        train_path_losses, train_channel_losses, config.train_K
    )

    test_config = WirelessConfig(config.test_K)
    test_coor, test_dists = generate_layouts_test(test_config, config.test_layouts)
    test_path_losses = compute_path_losses(test_config, test_dists)
    test_channel_losses = add_shadowing(test_path_losses)
    test_channel_losses = add_fast_fading(test_channel_losses)

    directLink_channel_losses = get_directLink_channel_losses(test_channel_losses)
    crossLink_channel_losses = get_crossLink_channel_losses(test_channel_losses)
    norm_train_dists, norm_test_dists = normalize_data(
        1 / train_dists,
        1 / test_dists,
        config.train_K,
        config.test_K,
        config.train_layouts,
        config.test_layouts,
    )
    norm_train_losses, norm_test_losses = normalize_data(
        np.sqrt(train_channel_losses),
        np.sqrt(test_channel_losses),
        config.train_K,
        config.test_K,
        config.train_layouts,
        config.test_layouts,
    )

    train_data_list = proc_data(
        train_losses,
        train_dists,
        norm_train_dists,
        norm_train_losses,
        config.train_K,
        config.graph_threshold,
    )
    test_data_list = proc_data(
        test_channel_losses,
        test_dists,
        norm_test_dists,
        norm_test_losses,
        config.test_K,
        config.graph_threshold,
    )
    train_loader = DataLoader(
        train_data_list,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.data_loader_workers,
    )
    test_loader = DataLoader(
        test_data_list,
        batch_size=config.test_layouts,
        shuffle=False,
        num_workers=config.data_loader_workers,
    )
    return {
        "train_config": train_config,
        "test_config": test_config,
        "var": var,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "test_channel_losses": test_channel_losses,
        "directLink_channel_losses": directLink_channel_losses,
        "crossLink_channel_losses": crossLink_channel_losses,
    }


def _train_and_record(model, name, datasets, config, device, results):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
    )
    train_model(
        model,
        name,
        datasets["train_loader"],
        datasets["test_loader"],
        optimizer,
        scheduler,
        device,
        config.train_K,
        config.test_K,
        datasets["var"],
        config.train_layouts,
        config.test_layouts,
        datasets["test_config"],
        datasets["directLink_channel_losses"],
        datasets["crossLink_channel_losses"],
        config.epochs,
        results,
    )


def run_baselines(datasets, config):
    Pini = np.random.rand(config.test_layouts, config.test_K, 1)
    Y1 = batch_WMMSE2_1(
        Pini,
        np.ones([config.test_layouts, config.test_K]),
        np.sqrt(datasets["test_channel_losses"]),
        1,
        datasets["var"],
    )
    Y2 = batch_WMMSE2(
        Pini,
        np.ones([config.test_layouts, config.test_K]),
        np.sqrt(datasets["test_channel_losses"]),
        1,
        datasets["var"],
    )

    rates1 = compute_rates(
        datasets["test_config"],
        Y1,
        datasets["directLink_channel_losses"],
        datasets["crossLink_channel_losses"],
    )
    rates2 = compute_rates(
        datasets["test_config"],
        Y2,
        datasets["directLink_channel_losses"],
        datasets["crossLink_channel_losses"],
    )

    sr1 = np.mean(np.sum(rates1, axis=1))
    sr2 = np.mean(np.sum(rates2, axis=1))

    print("FPlinQ fade:", sr1)
    print("WMMSE fade:", sr2)

    bl_Y = simple_greedy(
        datasets["test_channel_losses"],
        np.ones([config.test_layouts, config.test_K]),
        Y1,
        config.test_K,
    )
    rates_bl = compute_rates(
        datasets["test_config"],
        bl_Y,
        datasets["directLink_channel_losses"],
        datasets["crossLink_channel_losses"],
    )
    sr_bl = np.mean(np.sum(rates_bl, axis=1))
    print("baseline:", sr_bl)
    return {"sr_bl": sr_bl, "sr1": sr1, "sr2": sr2}


def print_results(results_GNN, baseline_results):
    for item in results_GNN:
        print(item)
    print("baseline:", round(baseline_results["sr_bl"], 4))
    print("WMMSE100 fade:", round(baseline_results["sr2"], 4))
    print("WMMSE_1 fade:", round(baseline_results["sr1"], 4))


if __name__ == "__main__":
    run_experiment()

