# -*- coding: utf-8 -*-

import numpy as np


def compute_SINRs(general_para, allocs, directlink_channel_losses, crosslink_channel_losses):
    assert np.shape(directlink_channel_losses) == np.shape(allocs), (
        "Mismatch shapes: {} VS {}".format(
            np.shape(directlink_channel_losses), np.shape(allocs)
        )
    )
    SINRs_numerators = allocs * directlink_channel_losses
    SINRs_denominators = (
        np.squeeze(np.matmul(crosslink_channel_losses, np.expand_dims(allocs, axis=-1)))
        + general_para.output_noise_power / general_para.tx_power
    )
    SINRs = SINRs_numerators / SINRs_denominators
    return SINRs


def compute_rates(general_para, allocs, directlink_channel_losses, crosslink_channel_losses):
    SINRs = compute_SINRs(
        general_para, allocs, directlink_channel_losses, crosslink_channel_losses
    )
    rates = np.log2(1 + SINRs)
    return rates

