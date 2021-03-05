#!/usr/bin/env python

import argparse
import os
import numpy as np
import logging
from tqdm import tqdm
from schnetpack import Properties
from schnetpack.md import MDUnits, HDF5Loader

from ase import Atoms, units, io, data

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def get_temperature():
    raise NotImplementedError


def write_trajectory(filename, atom_types, structures, cells=None, format='xyz', pbc=None):
    molecules = []
    for i in tqdm(range(len(structures)), ncols=120):
        if cells is None:
            ccell = None
        else:
            ccell = cells[i]
        atoms = Atoms(atom_types, structures[i], pbc=pbc, cell=ccell)
        atoms.wrap()
        molecules.append(atoms)
    io.write(filename, molecules, format=format)


def write_molecules(args):
    # Load the database
    database = HDF5Loader(args.hdf5_database)

    # Generate master directory
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    # Load metadata
    n_replicas = database.n_replicas
    n_molecules = database.n_molecules
    n_atoms = database.n_atoms
    atom_types = database.properties[Properties.Z]

    for molecule in range(n_molecules):
        logging.info('Extracting molecule {:d}...'.format(molecule + 1))

        # Get number of atoms and mask hdf5 dataset accordingly
        mol_n_atoms = n_atoms[molecule]
        mol_atom_types = atom_types[molecule, :mol_n_atoms]

        mol_structures = database.properties[Properties.position][:, :, molecule, ...] * MDUnits.internal2unit("Bohr")
        mol_velocities = database.properties["velocities"][:, :, molecule, ...]

        basename = os.path.join(args.target_dir, 'mol_{:03d}'.format(molecule + 1))

        # If requested compute the centroid of everything
        if args.centroid:
            mol_structures = np.mean(mol_structures, axis=1, keepdims=True)
            mol_velocities = np.mean(mol_velocities, axis=1, keepdims=True)
            n_replicas = 1
            basename += '_centroid'

        for replica in range(n_replicas):
            replica_atom_types = mol_atom_types
            masses = data.atomic_masses[replica_atom_types] * units._amu / units._me

            # If requested, also store velocities
            if args.temperature:
                filename = basename + '_temperature_replica_{:03d}.txt'.format(replica + 1)
                replica_velocities = mol_velocities[:, replica, ...]

                # Compute temperature
                kinetic_energy = 0.5 * np.sum(np.sum(replica_velocities ** 2, axis=2) * masses[None, :], axis=1)
                temperature = 2.0 / (3.0 * MDUnits.kB * mol_n_atoms) * kinetic_energy
                temperature_mean = np.mean(temperature)
                temperature_fluct = np.std(temperature)

                # For quick analysis
                logging.info('Replica {:3d}'.format(replica + 1))
                logging.info(' T_mean: {:10.3f}'.format(temperature_mean))
                logging.info(' T_std:  {:10.3f}'.format(temperature_fluct))

                np.savetxt(filename, temperature, fmt='%12.6f')
                logging.info('Stored temperature data to {:s}'.format(filename))

            else:
                filename = basename + '_replica_{:03d}.{:s}'.format(replica + 1, args.format)
                replica_structures = mol_structures[:, replica, ...] * units.Bohr

                write_trajectory(filename, replica_atom_types, replica_structures, format=args.format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_database', type=str, help='Path to HDF5 data written during MD.')
    parser.add_argument('target_dir', type=str, help='Directory to which to extract the geometries.')
    parser.add_argument('--temperature', action='store_true', help='Analyze the trajectory temperature.')
    parser.add_argument('--centroid', action='store_true', help='Compute centroids of structure.')
    parser.add_argument('--merge', action='store_true', help='Merge all replicas into one molecule')
    parser.add_argument('--format', type=str, choices=['xyz', 'pdb'], default='xyz', help='Format of geometry files.')
    parser.add_argument('--every', type=int, default=1, help='Write out every nth geometry')
    args = parser.parse_args()

    write_molecules(args)
