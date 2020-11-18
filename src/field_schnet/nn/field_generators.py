import torch
from schnetpack import nn as spknn
from torch import nn as nn

__all__ = [
    "ReactionField",
    "PointChargeField"
]


class ReactionField(nn.Module):
    """
    Modify external field through presence of an electrostatic continuum with a certain dielectric constant
    """

    def __init__(self, atom_features, radius=None):
        super(ReactionField, self).__init__()

        self.surface_factor = nn.Sequential(
            spknn.Dense(atom_features, atom_features, activation=spknn.shifted_softplus),
            spknn.Dense(atom_features, 1, activation=torch.sigmoid)
        )

        if radius is not None:
            self.radius = 1.0 / radius ** 3
        else:
            self.radius = 1.0

    def forward(self, x, mu, field, dielectric_constant):
        external_factor = (3 * dielectric_constant / (2 * dielectric_constant + 1))[:, None, None, :]
        internal_factor = (2 * (dielectric_constant - 1) / (2 * dielectric_constant + 1))[:, None, None, :]
        surface_factor = self.surface_factor(x)[:, :, :, None]

        external_field = external_factor * field
        internal_field = internal_factor * surface_factor * mu * self.radius

        reaction_field = internal_field + external_field

        return reaction_field


class PointChargeField(nn.Module):

    def __init__(self):
        super(PointChargeField, self).__init__()

    @staticmethod
    def forward(positions, charges, charge_positions):
        # Get field of point charges at every atom
        vectors = positions[:, None, :, :] - charge_positions[:, :, None, :]

        distances = torch.norm(vectors, 2, dim=3, keepdim=True)
        external_field = torch.sum(charges[:, :, None, None] * vectors / distances ** 3, 1)
        return external_field
