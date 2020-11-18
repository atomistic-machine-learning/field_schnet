import torch
from torch import nn as nn

import schnetpack as spk
from schnetpack import nn as spknn

from field_schnet.nn.basic import collect_neighbors
from field_schnet.nn.cutoff import MollifierCutoff

__all__ = [
    "DipoleLayer",
    "FieldInteraction",
    "TensorInteraction"
]


class DipoleLayer(nn.Module):

    def __init__(
            self,
            atom_features,
            dipole_features,
            transform=True,
            cutoff=None,
            activation=spk.nn.shifted_softplus
    ):
        super(DipoleLayer, self).__init__()
        if transform:
            self.transform = nn.Sequential(
                spk.nn.Dense(atom_features, atom_features, activation=activation),
                spk.nn.Dense(atom_features, dipole_features, activation=activation)
            )
        else:
            self.transform = None
        self.cutoff = cutoff

    def forward(self, x, r_ij, v_ij, neighbors, neighbor_mask):

        # Apply transformation layer
        if self.transform is not None:
            q = self.transform(x)
        else:
            q = x

        # Get neighbor contributions
        q_ij = collect_neighbors(q, neighbors)

        # Compute dipole moments
        mu_ij = q_ij[:, :, :, :, None] * v_ij[:, :, :, None, :] * neighbor_mask[:, :, :, None, None]

        # Apply cutoff if requested
        if self.cutoff is not None:
            c_ij = self.cutoff(r_ij)
            mu_ij = mu_ij * c_ij[:, :, :, None, None]

        # Form final sum
        mu = torch.sum(mu_ij, 2)

        return mu


class FieldInteraction(nn.Module):
    """
    Interaction with external field (magnetic, electric, ...)
    """

    def __init__(self, dipole_features, atom_features):
        super(FieldInteraction, self).__init__()
        self.dense = nn.Sequential(
            spknn.Dense(dipole_features, dipole_features, activation=spknn.shifted_softplus),
            spknn.Dense(dipole_features, atom_features)
        )

    def forward(self, mu, field):
        """
        Compute linear interaction with field
        """
        # For V x V, this is faster than matmul
        v = torch.sum(mu * field, 3)
        v = self.dense(v)
        return v


class TensorInteraction(nn.Module):

    def __init__(
            self,
            dipole_features,
            atom_features,
            cutoff=5.0,
            n_gaussians=50,
            shielding=None,
            cutoff_function=MollifierCutoff
    ):
        super(TensorInteraction, self).__init__()

        self.cutoff = cutoff_function(cutoff)
        self.shielding = shielding

        self.dense = nn.Sequential(
            spknn.Dense(dipole_features, dipole_features, activation=spknn.shifted_softplus),
            spknn.Dense(dipole_features, atom_features)
        )

        self.distance_expansion = nn.Sequential(
            spknn.Dense(n_gaussians, dipole_features, activation=spknn.shifted_softplus),
            spknn.Dense(dipole_features, dipole_features, bias_init=spknn.zeros_initializer)
        )

    def forward(self, mu, distances, distance_vector, neighbors, f_ij, neighbor_mask=None):
        # Get neighboring dipole moments
        B, A, F, X = mu.shape
        B, A, N = distances.shape
        mu_j = collect_neighbors(mu.view(B, A, -1), neighbors)
        mu_j = mu_j.view(B, A, N, F, X)

        # Matrix multiplication with interaction tensor expressed as dot products
        # mu_i @ Tij @ mu_j.T = mu_i @ ( -1 rij^2 + 3 Rij.T Rij ) @ mu_j
        #                     = -(mu_i @ mu_j.T)*rij^2 + 3 (mu_i @ Rij.T)*(mu_j @ Rij)
        diagonal_term = torch.sum(mu[:, :, None, :, :] * mu_j, dim=4) * distances[..., None] ** 2
        outer_term = torch.sum(mu[:, :, None, :, :] * distance_vector[:, :, :, None, :], dim=4) + \
                     torch.sum(mu_j * distance_vector[:, :, :, None, :], dim=4)

        # Inverse r^5 in this term
        radial = self.distance_expansion(f_ij) / distances[..., None] ** 5

        # Apply shielding
        if self.shielding is not None:
            radial = radial * (1 - self.shielding(distances)[..., None])

        if self.cutoff is not None:
            radial = radial * self.cutoff(distances)[..., None]

        if neighbor_mask is not None:
            radial = radial * neighbor_mask[..., None]

        v = (3 * diagonal_term - outer_term) * radial

        # Sum over neighbors
        v = torch.sum(v, 2)

        # Apply final transformation
        v = self.dense(v)

        return v
