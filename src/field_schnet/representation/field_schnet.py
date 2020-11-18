import torch
import torch.nn as nn
from ase import units

from schnetpack.representation import SchNetInteraction
from schnetpack import Properties
import schnetpack.nn as spknn

from field_schnet.nn.field_interactions import *
from field_schnet.nn.field_generators import ReactionField, PointChargeField
from field_schnet.nn import MollifierCutoff

__all__ = [
    "FieldSchNet"
]


class FieldSchNet(nn.Module):

    def __init__(self,
                 features=128,
                 interactions=5,
                 cutoff=5.0,
                 num_gaussians=25,
                 max_z=100,
                 cutoff_network=MollifierCutoff,
                 required_fields=(),
                 dipole_features=128,
                 dipole_cutoff=None,
                 field_mode="field",
                 ):
        super(FieldSchNet, self).__init__()

        self.n_interactions = interactions
        self.fields = required_fields
        self.field_mode = field_mode

        # atom type embeddings
        self.embedding = nn.Embedding(max_z, features, padding_idx=0)

        # spatial features
        self.distances = spknn.AtomDistances(return_directions=True)

        # Distance expansion
        self.distance_expansion = spknn.GaussianSmearing(0.0, cutoff, num_gaussians)

        # SchNet Interactions
        self.interactions = nn.ModuleList([
            SchNetInteraction(n_atom_basis=features,
                              n_spatial_basis=num_gaussians,
                              n_filters=features,
                              cutoff_network=cutoff_network,
                              cutoff=cutoff)
            for _ in range(interactions)
        ])

        # Set the cutoff for the dipole moment layers if requested above
        if dipole_cutoff is None:
            dipole_cutoff = cutoff
        dipole_cutoff = cutoff_network(dipole_cutoff)

        dipole_update = {}
        dipole_interactions = {}
        field_interactions = {}

        # Set up field structures
        for field in self.fields:
            if field not in Properties.external_fields:
                raise ValueError('Unrecognized field option {:s}'.format(field))

            # Generation of dipoles
            dipole_update[field] = nn.ModuleList([
                DipoleLayer(features, dipole_features,
                            transform=True,
                            cutoff=dipole_cutoff)
                for _ in range(interactions + 1)
            ])

            # Interactions between dipoles
            dipole_interactions[field] = nn.ModuleList([
                TensorInteraction(dipole_features, features,
                                  cutoff=cutoff,
                                  n_gaussians=num_gaussians)
                for _ in range(interactions)
            ])

            # Interaction with external fields
            field_interactions[field] = nn.ModuleList([
                FieldInteraction(dipole_features, features) for _ in range(interactions)])

            # Set up reaction field for implicit solvation

        self.dipole_update = nn.ModuleDict(dipole_update)
        self.dipole_interactions = nn.ModuleDict(dipole_interactions)
        self.field_interactions = nn.ModuleDict(field_interactions)

        # Field for implicit solvent
        if self.field_mode == "solvent":
            self.solvent_field = nn.ModuleList([
                ReactionField(features, radius=cutoff) for _ in range(interactions)
            ])
        else:
            self.solvent_field = None

        # Field for external point charges
        if self.field_mode == "qmmm":
            self.external_charge_field = PointChargeField()
        else:
            self.external_charge_field = None

    def forward(self, inputs):
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]

        # Apply field of external point charges
        if self.external_charge_field is not None:
            external_charge_field = self.external_charge_field(
                positions,
                inputs['external_charge_positions'],
                inputs['external_charges'],
            )

        # Spatial features
        r_ij, v_ij = self.distances(positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask)
        f_ij = self.distance_expansion(r_ij)

        # initial atom embedding features
        x = self.embedding(atomic_numbers)

        # Initial vector features (mu0 kept separate for derivatives)
        mu = {}
        mu0 = {}
        for field in self.fields:
            # mu0[field] = torch.zeros(x.shape + (3,), device=x.device)
            mu0[field] = self.dipole_update[field][0](x, r_ij, v_ij, neighbors, neighbor_mask)
            mu[field] = mu0[field]

        # Compute interactions
        for l in range(self.n_interactions):

            # Conventional SchNet update
            v_total = self.interactions[l](x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)

            # Field updates
            for field in self.fields:
                # Modify the external field if a implicit solvent is present
                if self.solvent_field is not None and field == Properties.electric_field:
                    effective_field = self.solvent_field[l](
                        x, mu[field], inputs[field], inputs[Properties.dielectric_constant]
                    )
                elif self.external_charge_field is not None and field == Properties.electric_field:
                    effective_field = inputs[field][:, None, None, :] + external_charge_field
                else:
                    effective_field = inputs[field][:, None, None, :]

                # Field interaction
                v_field = self.field_interactions[field][l](
                    mu[field], effective_field
                )

                # Dipole interaction
                v_dipoles = self.dipole_interactions[field][l](
                    mu[field], r_ij, v_ij, neighbors, f_ij=f_ij, neighbor_mask=neighbor_mask
                )

                v_total = v_total + v_field + v_dipoles

            # Update features
            x = x + v_total

            # Update dipole features
            for field in self.fields:
                mu[field] = mu[field] + self.dipole_update[field][l + 1](v_total, r_ij, v_ij, neighbors, neighbor_mask)

        return x, mu0
