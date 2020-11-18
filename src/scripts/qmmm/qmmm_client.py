#!/usr/bin/env python

import argparse
import os
import pickle
import socket
from schnetpack import Properties
from field_schnet.utils.qmmm_utils import NAMDData

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
HEADERSIZE = 20
OUT_STRING = 4 * '{:22.15f}' + '\n'


def write_output(results, namd_data):
    energy = results[Properties.energy]
    forces = results[Properties.forces]
    charges = results["charges"]

    with open(namd_data.output_filename, 'w') as ofile:
        ofile.write('{:22.15f}\n'.format(energy))
        for idx in range(namd_data.n_atoms):
            ofile.write(OUT_STRING.format(*forces[idx], charges[idx]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', type=str, help='Basename of NAMD input file')
    parser.add_argument('--port', type=int, default=65532, help="Port for socket.")
    args = parser.parse_args()

    # NAMD automatically changes to the QM directory ...
    input_filename = os.path.split(args.input_filename)[-1]

    # Read data from input file
    namd_data = NAMDData(input_filename)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, args.port))

        # Make pickle object and send
        msg = pickle.dumps(namd_data)
        msg = bytes(f"{len(msg):<{HEADERSIZE}}", "utf-8") + msg
        s.sendall(msg)

        # Receive results:
        full_msg = b''
        new_msg = True
        while True:
            msg = s.recv(2048)

            if new_msg:
                msglen = int(msg[:HEADERSIZE])
                new_msg = False

            full_msg += msg

            if len(full_msg) - HEADERSIZE == msglen:
                print("Full results received")
                results = pickle.loads(full_msg[HEADERSIZE:])
                break

        # Write to output file
        write_output(results, namd_data)
