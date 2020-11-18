import torch
import torch.nn as nn

from field_schnet.nn import general_derivative
from schnetpack import Properties
from field_schnet.nn.field_generators import PointChargeField


class FieldSchNetModel(nn.Module):

    def __init__(self, field_representation, energy_model, requested_properties=(), use_mean=False):
        super(FieldSchNetModel, self).__init__()

        self.requested_properties = requested_properties
        self.field_representation = field_representation
        self.energy_model = energy_model

        self.use_mean = use_mean

        # get computation instructions
        self.instructions = self._construct_properties()

        # Initialize the required derivatives
        self.grads_required = []

        # Store field mode for correct treatment of external charges
        self.field_mode = field_representation.field_mode

        # For compatibility with schnet model loading:
        self.output_modules = []

        # Settings of field for QM/MM
        # Field for external point charges
        if self.field_mode == "qmmm":
            self.external_charge_field = PointChargeField()
        else:
            self.external_charge_field = None

        for p in requested_properties:
            grads = Properties.required_grad[p]
            for grad in grads:
                if grad in Properties.external_fields and grad not in field_representation.fields:
                    raise ValueError('Field representation does not contain {:s} required for {:s}'.format(grad, p))
                else:
                    # Avoid duplicate entries
                    if grad not in self.grads_required:
                        self.grads_required.append(grad)

    def forward(self, inputs):
        # Set required derivatives
        for g in self.grads_required:
            if g == Properties.position:
                inputs[Properties.R].requires_grad = True
            if g == Properties.electric_field and g in inputs:
                inputs[Properties.electric_field].requires_grad = True
            if g == Properties.magnetic_field:
                inputs[Properties.magnetic_field].requires_grad = True

        # Apply field of external point charges if necessary
        if self.field_mode == "qmmm":
            if 'external_charge_positions' in inputs:
                inputs[Properties.electric_field] = self.external_charge_field(
                    inputs[Properties.R],
                    inputs['external_charges'],
                    inputs['external_charge_positions'],
                )

        # Get representation, inputs and mu0 if it is there.
        x, mu0 = self.field_representation(inputs)
        inputs['representation'] = x

        # Compute potential energy expression
        E = self.energy_model(inputs)['y']

        properties = {
            Properties.energy: E
        }

        # Do the derivatives as requested
        if self.instructions['dEdR']:
            # Compute forces
            dEdR = general_derivative(E, inputs[Properties.R])
            properties[Properties.forces] = -dEdR

            if self.instructions['d2EdR2']:
                # Compute the Hessian matrix
                d2EdR2 = general_derivative(dEdR, inputs[Properties.R], hessian=True)
                properties[Properties.hessian] = d2EdR2

        if self.instructions['dEdF']:

            # Compute the dipole moments
            dEdF = general_derivative(E, inputs[Properties.electric_field])
            if self.field_mode == "qmmm":
                dEdF = torch.sum(dEdF, 1)

            properties[Properties.dipole_moment] = -dEdF

            # Since a loop is used, it is more efficient to use dEdF and d2EdF2 as a basis and take dR!
            if self.instructions['d2EdFdR']:
                # Compute dipole derivatives
                d2EdRdF = general_derivative(dEdF, inputs[Properties.R])
                # Recover correct order
                d2EdRdF = d2EdRdF.permute(0, 2, 3, 1)
                properties[Properties.dipole_derivatives] = d2EdRdF

            if self.instructions['d2EdF2']:
                # Compute the polarizability tensor
                d2EdF2 = general_derivative(dEdF, inputs[Properties.electric_field])
                if self.field_mode == "qmmm":
                    d2EdF2 = torch.sum(d2EdF2, 2)

                properties[Properties.polarizability] = -d2EdF2

                if self.instructions['d3EdF2dR']:
                    # Compute the polarizability derivatives
                    d3EdRdF2 = general_derivative(d2EdF2, inputs[Properties.R])
                    # Recover proper order
                    d3EdRdF2 = d3EdRdF2.permute(0, 3, 4, 1, 2)
                    properties[Properties.polarizability_derivatives] = d3EdRdF2

        if self.instructions['dEdB']:
            # Compute magnetic moment
            dEdB = general_derivative(E, inputs[Properties.magnetic_field])
            properties['dEdB'] = dEdB

            # Compute shielding tensor
            if self.instructions['d2EdBdI']:
                d2EdBdI = general_derivative(dEdB, mu0[Properties.magnetic_field])

                if self.use_mean:
                    d2EdBdI = torch.mean(d2EdBdI, 3)
                else:
                    d2EdBdI = torch.sum(d2EdBdI, 3)

                d2EdBdI = d2EdBdI.permute(0, 2, 1, 3)
                properties[Properties.shielding] = d2EdBdI

        return properties

    def _construct_properties(self):
        instructions = {
            'dEdR': False,
            'd2EdR2': False,
            'dEdF': False,
            'd2EdFdR': False,
            'd2EdF2': False,
            'd3EdF2dR': False,
            'dEdB': False,
            'd2EdBdI': False
        }

        if (Properties.forces in self.requested_properties) or (
                Properties.hessian in self.requested_properties):
            instructions['dEdR'] = True

            if Properties.hessian in self.requested_properties:
                instructions['d2EdR2'] = True

        if (Properties.dipole_moment in self.requested_properties) or (
                Properties.polarizability in self.requested_properties) or (
                Properties.dipole_derivatives in self.requested_properties) or (
                Properties.polarizability_derivatives in self.requested_properties):

            instructions['dEdF'] = True

            if Properties.dipole_derivatives in self.requested_properties:
                instructions['d2EdFdR'] = True

            if (Properties.polarizability in self.requested_properties) or (
                    Properties.polarizability_derivatives in self.requested_properties):
                instructions['d2EdF2'] = True

            if Properties.polarizability_derivatives in self.requested_properties:
                instructions['d3EdF2dR'] = True

        if Properties.shielding in self.requested_properties:
            instructions['dEdB'] = True
            instructions['d2EdBdI'] = True

        return instructions
