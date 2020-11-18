#!/home/mitx/anaconda3/bin/python

import torch
import argparse
import os
import numpy as np
import logging
from ase import units, Atoms
import socket
import pickle

import field_schnet
from schnetpack import Properties
from schnetpack.environment import SimpleEnvironmentProvider

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
HEADERSIZE = 20

au2namd_energy = units.Ha / units.kcal * units.mol
au2namd_forces = au2namd_energy / units.Bohr


class QMMMDriver:

    def __init__(self):
        self.results = None

    def calculate(self, namd_data):
        self._create_input(namd_data)
        self._run_calculation()
        results = self._parse_output()
        return results

    def _create_input(self, namd_data):
        raise NotImplementedError

    def _run_calculation(self):
        raise NotImplementedError

    def _parse_output(self):
        raise NotImplementedError


class FieldSchNetDriver(QMMMDriver):

    def __init__(self, model_path, device=torch.device('cpu'),
                 environment_provider=SimpleEnvironmentProvider()):
        super(FieldSchNetDriver, self).__init__()

        self.device = device
        self.environment_provider = environment_provider

        # Load the model
        self.model = self._load_model(model_path)
        print('Loaded model...')

    def _load_model(self, model_path):
        """
        Load model and activate properties for atomic charges.
        """
        model = torch.load(model_path).to(self.device)

        requested_properties = model.requested_properties
        requested_properties += [Properties.dipole_derivatives]

        # Turn of shielding, as it is not required
        # requested_properties.remove(Properties.shielding)

        # model = field_schnet.model.PropertyModel(
        model = field_schnet.model.FieldSchNetModel(
            model.field_representation,
            model.energy_model,
            requested_properties=requested_properties
        )
        return model

    def _create_input(self, namd_data):

        self.inputs = {Properties.magnetic_field: torch.zeros(3)}

        # Pass external charges for computation of derivable field
        self.inputs['external_charge_positions'] = torch.FloatTensor(namd_data.charge_positions / units.Bohr)
        self.inputs['external_charges'] = torch.FloatTensor(namd_data.charges)

        atoms = Atoms(namd_data.atom_types, namd_data.positions / units.Bohr)

        # Elemental composition
        self.inputs[Properties.Z] = torch.LongTensor(atoms.get_atomic_numbers().astype(np.int))
        self.inputs[Properties.atom_mask] = torch.ones_like(self.inputs[Properties.Z]).float()

        # Set positions
        self.inputs[Properties.R] = torch.FloatTensor(atoms.positions)

        # get atom environment
        nbh_idx, offsets = self.environment_provider.get_environment(atoms)

        # Get neighbors and neighbor mask
        mask = torch.FloatTensor(nbh_idx) >= 0
        self.inputs[Properties.neighbor_mask] = mask.float()
        self.inputs[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int)) * mask.long()

        # Add batch dimension and move to CPU/GPU
        for key, value in self.inputs.items():
            self.inputs[key] = value.unsqueeze(0).to(self.device)

        self.inputs[Properties.cell] = None
        self.inputs[Properties.cell_offset] = None

    def _run_calculation(self):
        # TODO: Activate charge derivatives!!!!
        results = self.model(self.inputs)
        results['charges'] = self._compute_charges(results[Properties.dipole_derivatives])

        # Print dipole for IR
        shifts = torch.einsum("aii->a", results[Properties.shielding][0])
        print(("QMMM-SHIFTS:" + " {:20.11f}" * shifts.shape[0]).format(*shifts))
        print("QMMM-DIPOLE: {:20.11f} {:20.11f} {:20.11f}".format(*results[Properties.dipole_moment][0]))

        # Convert to numpy arrays
        self.results = {}
        for prop in results:
            self.results[prop] = results[prop].detach().cpu().numpy()

    def _parse_output(self):
        """
        Convert results to proper NAMD units
        """
        results = {}
        results[Properties.energy] = self.results[Properties.energy][0, 0] * au2namd_energy
        results[Properties.forces] = self.results[Properties.forces][0] * au2namd_forces
        results["charges"] = self.results['charges'][0]
        return results

    @staticmethod
    def _compute_charges(dipole_derivatives):
        charges = -torch.mean(torch.diagonal(dipole_derivatives, dim1=2, dim2=3), 2)
        return charges


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to ML model.')
    parser.add_argument('port', type=int, help="Port for socket.")
    parser.add_argument('max_connections', type=int, help="Maximum number of connections before server is shut down.")
    parser.add_argument('--cuda', action="store_true", help="Device to run computation on.")
    args = parser.parse_args()

    logging.info('Starting FieldSchNet server...')

    # Generate lock file
    open('qmmm_server.lock', 'a').close()

    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Set up the QM Driver
    driver = FieldSchNetDriver(args.model_path, device=device)

    # Perform the computation
    accepted_connections = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        s.bind((HOST, args.port))
        s.listen()

        # Remove lock file
        os.remove('qmmm_server.lock')

        while True:
            clientsocket, address = s.accept()

            with clientsocket:
                print(f"Calculation from {address} received.")

                full_msg = b''
                new_msg = True

                while True:
                    msg = clientsocket.recv(1024)
                    if new_msg:
                        msglen = int(msg[:HEADERSIZE])
                        new_msg = False

                    full_msg += msg

                    if len(full_msg) - HEADERSIZE == msglen:
                        namd_data = pickle.loads(full_msg[HEADERSIZE:])
                        break

                # 2) Perform prediction with driver
                print(f"Calculation started...")
                results = driver.calculate(namd_data)
                print(f"Calculation finished...")

                # 3) Send back data
                msg = pickle.dumps(results)
                msg = bytes(f"{len(msg):<{HEADERSIZE}}", "utf-8") + msg
                clientsocket.sendall(msg)

                # Increment accepted connections
                accepted_connections += 1

            if accepted_connections == args.max_connections:
                break
