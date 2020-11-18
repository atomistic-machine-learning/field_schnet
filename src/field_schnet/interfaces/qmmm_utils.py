import numpy as np


class NAMDData:

    def __init__(self, input_filename):

        # Set up basic file paths
        self.input_filename = input_filename
        self.output_filename = input_filename + '.result'

        # Generate data containers
        self.n_atoms = None
        self.n_charges = None

        self.atom_types = []
        self.positions = []

        self.charges = []
        self.charge_positions = []

        # Read in data from given file
        self._parse_namd_input(input_filename)

    def _parse_namd_input(self, input_filename):
        line_count = 0

        with open(input_filename, 'r') as infile:
            line = infile.readline().split()
            self.n_atoms = int(line[0])
            self.n_charges = int(line[1])

            for line in infile:
                line = line.split()

                if line_count < self.n_atoms:
                    self.atom_types.append(line[3])
                    self.positions.append([float(x) for x in line[0:3]])
                else:
                    self.charges.append(float(line[3]))
                    self.charge_positions.append([float(x) for x in line[0:3]])

                line_count += 1

        self.positions = np.array(self.positions)
        self.charges = np.array(self.charges)
        self.charge_positions = np.array(self.charge_positions)