#!/usr/bin/env python

import argparse
import os
import numpy as np
import logging
import matplotlib.pyplot as plt

import schnetpack as spk
from field_schnet.md.utils import NMRSpectrum, RamanSpectrum
from schnetpack.md.utils import IRSpectrum, PowerSpectrum

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def plot_spectrum(spectrum, name, max_freq=4000, upper=None):
    frequencies, intensities = spectrum
    plt.figure()
    plt.plot(frequencies, intensities)
    plt.title(name)
    plt.xlim(0, max_freq)
    plt.ylabel('I [a.u.]')
    plt.xlabel('$\omega$ [cm$^{-1}$]')
    if upper is not None:
        plt.ylim(0, upper)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_database', type=str, help='Path to HDF5 data written during MD.')
    parser.add_argument('spectrum_file', type=str, help='File to which the spectra are stored.')
    parser.add_argument('--resolution', type=int, default=4096, help='Spectrum resolution.')
    parser.add_argument('--skip', type=int, default=0, help='Skip the first N steps in the MD.')
    parser.add_argument('--mol_idx', type=int, default=0, help='Index of molecule to be loaded')

    # Type
    parser.add_argument('--spectrum', nargs='+', type=str, choices=['vdos', 'ir', 'raman', 'nmr', 'all'],
                        default='vdos', help='Types of spectra to compute. Default is vdos')

    # Plotting
    parser.add_argument('--plot', action='store_true', help='Plot spectra for inspection.')

    # Raman specific terms
    parser.add_argument('--raman_freq', type=float, default=19455.25,
                        help='Raman laser frequency, default' '=19455.25 cm^-1 (514 nm).')
    parser.add_argument('--raman_temp', type=float, default=300.0, help='Temperature for Raman spectrum.')
    parser.add_argument('--raman_average', action='store_true', help='Compute rotationally averaged Raman spectrum.')

    args = parser.parse_args()

    logging.info('Reading dataset {:s}'.format(args.hdf5_database))

    dataset = spk.md.utils.HDF5Loader(args.hdf5_database, skip_initial=args.skip)

    spectra = {}

    if 'vdos' in args.spectrum or 'all' in args.spectrum:
        power_spectrum = PowerSpectrum(dataset, resolution=args.resolution)
        power_spectrum.compute_spectrum(args.mol_idx)
        spectrum_vdos = power_spectrum.get_spectrum()

        # For storage
        spectra['vdos'] = spectrum_vdos

        if args.plot:
            plot_spectrum(spectrum_vdos, 'VDOS')

    if 'ir' in args.spectrum or 'all' in args.spectrum:
        ir_spectrum = IRSpectrum(dataset, resolution=args.resolution)
        ir_spectrum.compute_spectrum(args.mol_idx)
        spectrum_ir = ir_spectrum.get_spectrum()

        spectra['ir'] = spectrum_ir

        if args.plot:
            plot_spectrum(spectrum_ir, 'IR')

    if 'raman' in args.spectrum or 'all' in args.spectrum:
        raman_spectrum = RamanSpectrum(dataset, args.raman_freq, temperature=args.raman_temp,
                                       averaged=args.raman_average, resolution=args.resolution)
        raman_spectrum.compute_spectrum(args.mol_idx)
        spectrum_isotropic, spectrum_anisotropic = raman_spectrum.get_spectrum()

        spectrum_depol = (spectrum_isotropic[0], spectrum_anisotropic[1] / spectrum_isotropic[1])
        spectrum_depol[1][0] = 0.0

        spectra['raman'] = spectrum_isotropic
        spectra['raman_depol'] = spectrum_depol

        if args.plot:
            plot_spectrum(spectrum_isotropic, 'Raman', upper=0.03)
            plot_spectrum(spectrum_depol, 'Raman (depol)')

    if 'nmr' in args.spectrum or 'all' in args.spectrum:
        nmr_spectrum = NMRSpectrum(dataset)
        nmr_spectrum.compute_spectrum(args.mol_idx)
        pass

    # Store spectra
    file_name = args.spectrum_file

    if os.path.splitext(file_name)[1] != '.npz':
        file_name += '.npz'

    logging.info('Storing spectrum to {:s}'.format(file_name))
    np.savez(file_name, **spectra)
    plt.show()
