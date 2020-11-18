import torch

import schnetpack as spk
from schnetpack.md.calculators import MDCalculator

__all__ = [
    "FieldSchNetCalculator"
]


class FieldSchNetCalculator(MDCalculator):

    def __init__(
            self,
            model,
            required_properties,
            force_handle="forces",
            position_conversion=1.0,
            force_conversion=1.0,
            property_conversion={},
            detach=True
    ):
        super(FieldSchNetCalculator, self).__init__(
            required_properties,
            force_handle,
            position_conversion,
            force_conversion,
            property_conversion,
            detach=detach
        )

        self.model = model

        # Check which fields are present
        if not self._check_is_parallel():
            if hasattr(self.model, 'field_representation'):
                self.fields = [field for field in self.model.field_representation.fields]
            else:
                self.fields = []
        else:
            # Additional check for DataParallel
            if hasattr(self.model.module, 'field_representation'):
                self.fields = [field for field in self.model.module.field_representation.fields]
            else:
                self.fields = []

    def calculate(self, system):
        inputs = self._generate_input(system)
        self.results = self.model(inputs)
        self._update_system(system)

    def _generate_input(self, system):
        positions, atom_types, atom_masks = self._get_system_molecules(system)
        neighbors, neighbor_mask = self._get_system_neighbors(system)

        inputs = {
            spk.Properties.R: positions,
            spk.Properties.Z: atom_types,
            spk.Properties.atom_mask: atom_masks,
            spk.Properties.cell: None,
            spk.Properties.cell_offset: None,
            spk.Properties.neighbors: neighbors,
            spk.Properties.neighbor_mask: neighbor_mask
        }

        # Construct auxiliary field inputs
        n_batch = positions.shape[0]
        for field in self.fields:
            inputs[field] = torch.zeros(n_batch, 3, device=system.device)

        return inputs

    def _check_is_parallel(self):
        return True if isinstance(self.model, torch.nn.DataParallel) else False
