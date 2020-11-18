import json
import logging
import copy

import torch
from schnetpack import Properties


def compute_shielding_scaling(data_loader, device, scaling_target=1):
    """
    Compute scaling used in NMR shielding tensor loss.
    """
    scaling = {}
    scaling_count = {}

    for batch in data_loader:
        z = batch[Properties.Z]
        shielding = torch.einsum("baii->ba", batch[Properties.shielding]) / 3.0

        for z_idx in torch.unique(z):
            z_idx = int(z_idx.cpu().numpy())
            shielding_z = shielding[z == z_idx]

            if z_idx in scaling:
                scaling[z_idx] += torch.sum(shielding_z ** 2)
                scaling_count[z_idx] += shielding_z.numel()
            else:
                scaling[z_idx] = torch.sum(shielding_z ** 2)
                scaling_count[z_idx] = shielding_z.numel()

    for z in scaling_count:
        scaling[z] = scaling[z] / scaling_count[z]

    scaling_h = copy.deepcopy(scaling[scaling_target])
    for z in scaling:
        scaling[z] /= scaling_h
        logging.info("Loss scaling nucleus {:3d}: {:10.3f}".format(z, 1.0 / scaling[z]))

    max_z = max(scaling) + 1
    shielding_weights = torch.zeros(max_z, device=device)

    for z in scaling:
        shielding_weights[z] = 1.0 / scaling[z]

    return shielding_weights
