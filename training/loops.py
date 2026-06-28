# -*- coding: utf-8 -*-

import time

import numpy as np
import torch

from metrics.losses import sr_loss
from metrics.rates import compute_rates


def train_one_epoch(model, train_loader, optimizer, device, link_count, var, train_layouts):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = sr_loss(data, out, link_count, var, device)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / train_layouts


def evaluate(
    model,
    test_loader,
    device,
    link_count,
    var,
    test_layouts,
    test_config,
    directLink_channel_losses,
    crossLink_channel_losses,
):
    model.eval()
    total_loss = 0
    out = None
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            start = time.time()
            out = model(data)
            end = time.time()
            dnn_time = end - start
            print("dnn time", end - start)
            loss = sr_loss(data, out, link_count, var, device)
            total_loss += loss.item() * data.num_graphs

    power = out
    power = torch.reshape(power, (-1, link_count))
    Y = power.cpu().numpy()
    rates = compute_rates(
        test_config, Y, directLink_channel_losses, crossLink_channel_losses
    )
    sr = np.mean(np.sum(rates, axis=1))
    print("actual_rates:", sr)
    return total_loss / test_layouts


def train_model(
    model,
    model_name,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    device,
    train_K,
    test_K,
    var,
    train_layouts,
    test_layouts,
    test_config,
    directLink_channel_losses,
    crossLink_channel_losses,
    epochs,
    results,
):
    for epoch in range(1, epochs + 1):
        loss1 = train_one_epoch(
            model, train_loader, optimizer, device, train_K, var, train_layouts
        )
        if epoch % 10 == 0:
            loss2 = evaluate(
                model,
                test_loader,
                device,
                test_K,
                var,
                test_layouts,
                test_config,
                directLink_channel_losses,
                crossLink_channel_losses,
            )
            print(
                "Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}".format(
                    epoch, loss1, loss2
                )
            )
        if epoch % 100 == 0:
            results.append((model_name, round(loss1, 4), round(loss2, 4)))
        scheduler.step()
    return results

