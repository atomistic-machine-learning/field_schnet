# FieldSchNet - Deep neural network for molecules in external fields

FieldSchNet provides a deep neural network for modeling the interaction of molecules and external environments as
described in [1].
The package builds on the SchNetPack infrastructure [2] and provides functionality for training and deploying FieldSchNet
models for simulating molecular spectra and reactions in the presence of fields, continuum solvents, as well
as in a QM/MM setup.

##### Requirements:
- python 3
- torch>=0.4.1
- numpy
- ASE
- Hydra
- schnetpack>=0.3.0
- PyTorch (>=0.4.1)
- Optional: tensorboardX

_**Note: We recommend using a GPU for training the neural networks.**_

## Installation 

Clone the repository:
```
git clone git@github.com:atomistic-machine-learning/field_schnet.git
cd field_schnet
```

Install requirements:
```
pip install -r requirements.txt
```

Install FieldSchNet:
```
pip install .
```

## Example

Here, we show how to train a basic FieldSchNet model for predicting energies, forces, dipole moments and polarizability
tensors using the ethanol molecule as an example.
In addition, we demonstrate how a trained model can be used in molecular dynamics simulations to compute infrared and
Raman spectra.

All FieldSchNet scripts used in the example are inserted into your PATH during installation. 

### Training the model

A reference dataset `ethanol_vacuum.db` in ASE db format  (see [1] for details on the method) can be found in
the `example` directory.

A FieldSchNet model can be trained on this dataset via
```
field_schnet_run.py data_path=<PATH/TO/>ethanol_vacuum.db basename=<modeldir> cuda=true
```
where `data_path` should point to the reference data set. `basename` indicates the model directory and the `cuda=true`
flag activates GPU training.
The training progress will be logged in `<modeldir>/log`, either as CSV (default) or as TensorBoard event files.
A training run using default settings should take approximately five hours on a notebook GPU with 2 GB VRAM.

To evaluate the trained model with the best validation error, call
```
field_schnet_run.py data_path=<PATH/TO/>ethanol_vacuum.db basename=<modeldir> cuda=true mode=eval
```
which will run on the test set and write a result file `evaluation.txt` into the model directory.
The best model is stored in the file `best_model` in the same directory.


### Performing molecular dynamics and computing spectra

Once a model has been trained for ethanol, it can be used to simulate various molecular spectra (a pre-trained model can
be found under `example/ethanol_vacuum.model`). 

#### MD

We run a molecular dynamics (MD) simulation using the `md` module of SchNetPack (more details can be found in the
SchNetPack [MD tutorial](https://schnetpack.readthedocs.io/en/stable/tutorials/tutorial_04_molecular_dynamics.html)).

A basic input file template `md_input.yaml` for using FieldSchNet in conjunction with SchNetPack MD is provided in the 
`example` directory. 
To run a simulation, a few adaptations to this file are necessary:
- The `model_file` entry in the `calculator` block must be changed to a valid path to a trained model 
(e.g. `<modeldir>/best_model` or `example/ethanol_vacuum.model`)
- A path to a xyz-file containing an initial ethanol structure must be set in `molecule_file` (`system` block). The 
`example` directory contains a suitable ethanol structure (`ethanol_initial.xyz`)
- The `simulation_dir` placeholder should be changed to a reasonable name for the experiment.

The simulation is started with
`spk_md.py md_input.yaml`
It will generate the directory specified in `simulation_dir` and store the results of the simulation there.
MD related data (such as forces, properties and velocities) are stored in an hdf5 file
(`<simulation_dir>/simulation.hdf5`).

In general, data can be extracted using the `HDF5Loader` utility provided with SchNetPack
(`schnetpack.md.utils.hdf5_data`).
Here we provide the script `field_schnet_extract_hdf5.py` which can be used to convert the sampled structures to
XYZ-format
```
field_schnet_extract_hdf5.py <simulation_dir>/simulation.hdf5 <xyz_directory>
```
This will generate a trajectory in XYZ-format in the `<xyz_directory>`.

#### Spectra

Once a simulation has been performed, molecular spectra can be computed from a `simulation.hdf5` file with the
`field_schnet_spectra_hdf5.py` script.

To compute, store and plot spectra based on the above MD run, execute:
```
field_schnet_spectra_hdf5.py <simulation_dir>/simulation.hdf5 <spectrum.npz> --spectra ir raman --plot --skip_initial 10000
```
This will generate infrared and polarized and depolarized Raman spectra and plot them to the screen. The spectrum data
will also be stored to the `<spectrum.npz>` file. `--skip_initial 10000` indicates, that the first 10000 steps (5 ps) of
the trajectory should be seen as equilibration period and be skipped.

Please refer to `field_schnet_spectra_hdf5.py --help` for more details.

## Changing FieldSchNet settings

FieldSchNet uses [hydra](https://github.com/facebookresearch/hydra) for managing experiment configs. The default
settings produce a relatively small FieldSchNet model for demonstration purposes. These settings can be modified via
standard hydra syntax using the configurations defined in `src/scripts/configs`. The currently used config can also 
be printed via
```
field_schnet_run.py --cfg job
```
and optionally be stored to a file and modified. Such a configuration file can then be used in an experiment with the
command
```
field_schnet_run.py load_config=<PATH/TO/CONFIG>
```
which will override all changed default settings.

### Specifying properties

The properties fit by FieldSchNet are controlled via the `tradeoff` block. Properties can be added and removed by
changing the entries. Different pre-defined settings are available and can be changed by adding `tradeoffs=<setting>` to
the command line. E.g. changing the training command to
```
field_schnet_run.py data_path=<PATH/TO/>ethanol_vacuum.db basename=<modeldir> cuda=true tradeoffs=electromagnetic
```
will also include NMR shielding tensors during model training.

## References

* [1] M. Gastegger, K.T. Schütt, K.-R. Müller.
*Machine learning of solvent effects on molecular spectra and reactions*
(2020) https://arxiv.org/abs/2010.14942

* [2] K.T. Schütt, P. Kessel, M. Gastegger, K. Nicoli, A. Tkatchenko, K.-R. Müller.
*SchNetPack: A Deep Learning Toolbox For Atomistic Systems.*
J. Chem. Theory Comput, 15(1), 448–455 (2018) [10.1021/acs.jctc.8b00908](http://dx.doi.org/10.1021/acs.jctc.8b00908)
[arXiv:1809.01072](https://arxiv.org/abs/1809.01072)

