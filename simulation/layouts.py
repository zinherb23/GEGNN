# -*- coding: utf-8 -*-

import numpy as np


def generate_layouts(general_para, number_of_layouts):
    return _generate_layouts(general_para, number_of_layouts, layout_generate)


def generate_layouts_test(general_para, number_of_layouts):
    return _generate_layouts(general_para, number_of_layouts, layout_generate_test)


def layout_generate(general_para):
    return _layout_generate(
        general_para,
        general_para.shortest_directLink_length,
        general_para.longest_directLink_length,
    )


def layout_generate_test(general_para):
    return _layout_generate(
        general_para,
        general_para.shortest_directLink_length_test,
        general_para.longest_directLink_length_test,
    )


def _generate_layouts(general_para, number_of_layouts, layout_generator):
    N = general_para.n_links
    print(
        "<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(
            number_of_layouts, general_para.setting_str
        )
    )
    layouts = []
    dists = []
    for i in range(number_of_layouts):
        layout, dist = layout_generator(general_para)
        layouts.append(layout)
        dists.append(dist)
    layouts = np.array(layouts)
    dists = np.array(dists)
    assert np.shape(layouts) == (number_of_layouts, N, 4)
    assert np.shape(dists) == (number_of_layouts, N, N)
    return layouts, dists


def _layout_generate(general_para, min_direct_link_length, max_direct_link_length):
    N = general_para.n_links
    tx_xs = np.random.uniform(low=0, high=general_para.field_length, size=[N, 1])
    tx_ys = np.random.uniform(low=0, high=general_para.field_length, size=[N, 1])

    while True:
        rx_xs, rx_ys = _generate_receivers(
            general_para, tx_xs, tx_ys, min_direct_link_length, max_direct_link_length
        )
        layout = np.concatenate((tx_xs, tx_ys, rx_xs, rx_ys), axis=1)
        distances = _compute_distances(layout, N)
        if np.min(distances) > general_para.shortest_crossLink_length:
            break
    return layout, distances


def _generate_receivers(
    general_para, tx_xs, tx_ys, min_direct_link_length, max_direct_link_length
):
    rx_xs = []
    rx_ys = []
    for i in range(general_para.n_links):
        rx_x, rx_y = _generate_valid_receiver(
            general_para,
            tx_xs[i],
            tx_ys[i],
            min_direct_link_length,
            max_direct_link_length,
        )
        rx_xs.append(rx_x)
        rx_ys.append(rx_y)
    return rx_xs, rx_ys


def _generate_valid_receiver(
    general_para, tx_x, tx_y, min_direct_link_length, max_direct_link_length
):
    got_valid_rx = False
    while not got_valid_rx:
        pair_dist = np.random.uniform(
            low=min_direct_link_length, high=max_direct_link_length
        )
        pair_angles = np.random.uniform(low=0, high=np.pi * 2)
        rx_x = tx_x + pair_dist * np.cos(pair_angles)
        rx_y = tx_y + pair_dist * np.sin(pair_angles)
        if 0 <= rx_x <= general_para.field_length and 0 <= rx_y <= general_para.field_length:
            got_valid_rx = True
    return rx_x, rx_y


def _compute_distances(layout, n_links):
    distances = np.zeros([n_links, n_links])
    for rx_index in range(n_links):
        for tx_index in range(n_links):
            tx_coor = layout[tx_index][0:2]
            rx_coor = layout[rx_index][2:4]
            distances[rx_index][tx_index] = np.linalg.norm(tx_coor - rx_coor)
    return distances

