import numpy as np
from ase import units
from matplotlib import pyplot as plt
from schnetpack.md import VibrationalSpectrum, MDUnits


def lorentz_convolution(data, start, stop, n_points, width):
    offset = np.linspace(start, stop, n_points)
    spectrum = 1 / (1 + ((data[:, None] - offset[None, :]) / width) ** 2)
    return offset, np.sum(spectrum, axis=0)


class NMRSpectrum:
    tms_reference = {
        'pbe0/def2-TZVP': {
            1: 31.766,
            6: 188.530,
        }
    }
    au2ppm = units.alpha ** 2 * 1e6

    def __init__(self, data, reference='pbe0/def2-TZVP', first_n=10000):
        self.data = data
        self.reference = self.tms_reference[reference]
        self.first_n = first_n

    def compute_spectrum(self, molecule_idx):
        relevant_data = self._get_data(molecule_idx)
        atom_types = self.data.atom_types[molecule_idx]

        unique_elements = np.unique(atom_types)
        for element in unique_elements:
            if element in self.reference:
                element_data = self.reference[element] - \
                               relevant_data[:self.first_n, atom_types == element].flatten() * self.au2ppm
                if element == 1:
                    element_data = element_data[:self.first_n]
                    f, spec = lorentz_convolution(element_data, 0, 15, 5000, 0.1)
                if element == 6:
                    element_data = element_data[:self.first_n]
                    f, spec = lorentz_convolution(element_data, 0, 300, 5000, 1.0)
                plt.figure()
                plt.title(f'NMR {element}')
                plt.plot(f, spec)

    def _get_data(self, molecule_idx):
        relevant_data = self.data.get_property('shielding', mol_idx=molecule_idx)
        relevant_data = np.trace(relevant_data, axis1=2, axis2=3) / 3
        return relevant_data


class RamanSpectrum(VibrationalSpectrum):

    def __init__(self, data, incident_frequency, temperature=300, resolution=4096, averaged=False):
        super(RamanSpectrum, self).__init__(data, resolution=resolution)
        self.incident_frequency = incident_frequency
        self.temperature = temperature
        self.averaged = averaged

    def _get_data(self, molecule_idx):
        relevant_data = self.data.get_property('polarizability', mol_idx=molecule_idx)

        # Compute numerical derivative via central differences
        relevant_data = (relevant_data[2:, ...] - relevant_data[:-2, ...]) / (2 * self.timestep)

        # Compute isotropic and anisotropic part
        if self.averaged:
            # Setup for random orientations of the molecule
            polar_data = np.zeros((relevant_data.shape[0], 7))
            # Isotropic contribution:
            polar_data[:, 0] = np.trace(relevant_data, axis1=1, axis2=2) / 3
            # Anisotropic contributions
            polar_data[:, 1] = relevant_data[..., 0, 0] - relevant_data[..., 1, 1]
            polar_data[:, 2] = relevant_data[..., 1, 1] - relevant_data[..., 2, 2]
            polar_data[:, 3] = relevant_data[..., 2, 2] - relevant_data[..., 0, 0]
            polar_data[:, 4] = relevant_data[..., 0, 1]
            polar_data[:, 5] = relevant_data[..., 0, 2]
            polar_data[:, 6] = relevant_data[..., 1, 2]
        else:
            polar_data = np.zeros((relevant_data.shape[0], 2))
            # Typical experimental setup
            # xx
            polar_data[:, 0] = relevant_data[..., 0, 0]
            # xy
            polar_data[:, 1] = relevant_data[..., 0, 1]

        return polar_data

    def _process_autocorrelation(self, autocorrelation):
        if self.averaged:
            isotropic = autocorrelation[0, :]
            anisotropic = 0.5 * autocorrelation[1, :] + \
                          0.5 * autocorrelation[2, :] + \
                          0.5 * autocorrelation[3, :] + \
                          3.0 * autocorrelation[4, :] + \
                          3.0 * autocorrelation[5, :] + \
                          3.0 * autocorrelation[6, :]
        else:
            isotropic = autocorrelation[0, :]
            anisotropic = autocorrelation[1, :]

        autocorrelation = [isotropic, anisotropic]

        return autocorrelation

    def _process_spectrum(self):
        frequencies = self.frequencies[0]
        cross_section = (self.incident_frequency - frequencies) ** 4 / frequencies / (
                1 - np.exp(-(MDUnits.h_bar2icm * frequencies) / (MDUnits.kB * self.temperature)))
        cross_section[0] = 0

        for i in range(len(self.intensities)):
            self.intensities[i] *= cross_section
            self.intensities[i] *= 4.160440e-18
            self.intensities[i][0] = 0.0

        if self.averaged:
            isotropic, anisotropic = self.intensities
            parallel = (isotropic + 4 / 45 * anisotropic)
            orthogonal = anisotropic / 15

            self.intensities = [parallel, orthogonal]
