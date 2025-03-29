# parcellate

This codebase provides a command-line interface for functional brain parcellation of volumetric
fMRI data. An analysis is specified with a YAML configuration file (including paths to functional
data and evaluation task maps, if desired), which is passed as input to command-line utilities
for training and visualization of results. A minimal example is provided here (`example.yml`).

This README provides a quickstart introduction to the command-line interface. Detailed documentation
is available at
[https://parcellate.readthedocs.io/en/latest/#](https://parcellate.readthedocs.io/en/latest/#).

## Installation

Installation is just a matter of setting up an [Anaconda](https://www.anaconda.com/) environment
with the right software dependencies. Once you have Anaconda installed, you can create the
environment by running the following command in the terminal:

```bash
conda env create -f conda.yml
```

This will create a new environment called `parcellate` with all the necessary dependencies,
which you can activate by running:

```bash
conda activate parcellate
```

## Usage

This codebase currently provides two command-line utilities: `train` (for fitting a parcellation
for a new participant) and `plot` (for visualizing the results of a parcellation).

To train a new parcellation for a config file `example.yml`, run:

```bash
python -m parcellate.train example.yml <ARGS>
```

To visualize the results of a parcellation for a config file `example.yml`, run:

```bash
python -m parcellate.plot example.yml <ARGS>
```

Detailed usage for each utility can be viewed by running it with the `--help` flag.