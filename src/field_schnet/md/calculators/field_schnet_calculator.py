import torch

import schnetpack as spk
from schnetpack.md.calculators import MDCalculator, SchnetPackCalculator
from schnetpack.md.neighbor_lists import SimpleNeighborList, TorchNeighborList
import logging

__all__ = [
    "FieldSchNetCalculator"
]


class FieldSchNetCalculator(SchnetPackCalculator):

    def __init__(
            self,
            model,
            required_properties,
            force_handle="forces",
            position_conversion="Bohr",
            force_conversion="Ha/Bohr",
            property_conversion={},
            detach=True,
            neighbor_list=TorchNeighborList,
            cutoff=-1.0,
            cutoff_shell=1.0,
    ):
        model.representation = None
        super(FieldSchNetCalculator, self).__init__(
            model,
            required_properties,
            force_handle=force_handle,
            position_conversion=position_conversion,
            force_conversion=force_conversion,
            property_conversion=property_conversion,
            stress_handle=None,
            detach=detach,
            neighbor_list=neighbor_list,
            cutoff=cutoff,
            cutoff_shell=cutoff_shell,
            cutoff_lr=None,
        )

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
        positions, atom_types, atom_masks, cells, pbc = self._get_system_molecules(system)
        neighbors = self._get_system_neighbors(system)

        inputs = {
            spk.Properties.R: positions,
            spk.Properties.Z: atom_types,
            spk.Properties.atom_mask: atom_masks,
            spk.Properties.cell: cells,
            spk.Properties.pbc: pbc,
        }
        inputs.update(neighbors)

        # Construct auxiliary field inputs
        n_batch = positions.shape[0]
        for field in self.fields:
            inputs[field] = torch.zeros(n_batch, 3, device=system.device)

        return inputs

    def _check_is_parallel(self):
        return True if isinstance(self.model, torch.nn.DataParallel) else False

    def _get_model_cutoff(self):
        """
        Function to check the model passed to the calculator for already set cutoffs.
        Returns:
            float: Model cutoff in model position units.
        """
        # Get representation
        if hasattr(self.model, "module"):
            representation = self.model.module.field_representation
        else:
            representation = self.model.field_representation

        try:
            model_cutoff = representation.interactions[0].cutoff_network.cutoff
        except:
            raise ValueError("Could not find model cutoff, please specify in input file.")

        # Convert from torch tensor and print out the detected cutoff
        model_cutoff = float(model_cutoff[0].cpu().numpy())
        logging.info("Detected cutoff radius of {:5.3f}...".format(model_cutoff))

        return model_cutoff
