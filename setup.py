import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8') as f:
        return f.read()


setup(
    name='field_schnet',
    version='0.1.0',
    author='',
    packages=find_packages('src'),
    scripts=[
        'src/scripts/field_schnet_run.py',
        'src/scripts/field_schnet_spectra_hdf5.py',
        'src/scripts/field_schnet_extract_hdf5.py',
    ],
    package_dir={'': 'src'},
    python_requires='>=3.6',
    install_requires=[
        "schnetpack>=0.3.0"
        "torch>=0.4.1",
        "numpy",
        "ase>=3.16",
        "hydra>=1.0.0"
    ]
)
