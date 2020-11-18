from schnetpack import Properties
import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes

from schnetpack.interfaces import SpkCalculator


class FieldSchNetCalculator(SpkCalculator):
    implemented_properties = [Properties.energy, Properties.forces, "dipole", "shielding", "polarizability"]

    def __init__(
            self,
            model,
            device="cpu",
            energy_units="Ha",
            forces_units="Ha/Bohr",
            **kwargs
    ):
        model = torch.load(model, map_location=device).to(device)
        super(FieldSchNetCalculator, self).__init__(
            model=model,
            device=device,
            collect_triples=False,
            energy=Properties.energy,
            forces=Properties.forces,
            energy_units=energy_units,
            forces_units=forces_units,
            **kwargs
        )
        self.device = device

    def calculate(
            self,
            atoms=None,
            properties=(Properties.energy, Properties.forces, Properties.dipole_moment),
            system_changes=all_changes
    ):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): Properties to calculate.
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        if self.calculation_required(atoms, properties):

            Calculator.calculate(self, atoms, properties, system_changes)

            # Convert to schnetpack input format
            model_inputs = self.atoms_converter(atoms)
            # Convert to Bohr
            model_inputs["_positions"] /= units.Bohr

            # Add missing components for field SchNet
            # Construct auxiliary field inputs
            for field in self.model.field_representation.fields:
                model_inputs[field] = torch.zeros(1, 3, device=self.device)

            # Call model
            model_results = self.model(model_inputs)

            # Convert outputs to ASE calculator format
            self.results = {}
            if Properties.energy in properties:
                energy = model_results[Properties.energy].cpu().data.numpy()
                self.results[Properties.energy] = energy.reshape(-1) * units.Ha
            if Properties.forces in properties:
                forces = model_results[Properties.forces].cpu().data.numpy()
                self.results[Properties.forces] = forces.reshape((len(atoms), 3)) * units.Ha / units.Bohr
            if "dipole" in properties:
                dipole_moment = model_results[Properties.dipole_moment][0].cpu().data.numpy()
                self.results["dipole"] = dipole_moment.reshape(3) * units.Bohr
            if Properties.polarizability in properties:
                polar = model_results[Properties.polarizability][0].cpu().data.numpy()
                self.results[Properties.polarizability] = polar * units.Bohr ** 3
            if Properties.shielding in properties:
                shielding = model_results[Properties.shielding][0].cpu().data.numpy()
                self.results[Properties.shielding] = shielding
