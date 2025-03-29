.. parcellate documentation master file, created by
   sphinx-quickstart on Fri Mar 28 21:48:11 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

parcellate documentation
========================

This codebase provides a command-line interface for functional brain parcellation of volumetric
fMRI data. An analysis is specified with a YAML configuration file (including paths to functional
data and evaluation task maps, if desired), which is passed as input to command-line utilities
for training and visualization of results. A minimal example is provided in the repository
root (`example.yml`).

This documentation page is focused primarily on the internal API for parcellate. Typical usage
will not require delving into these details, but they are provided for reference. In most cases
the command-line usage (documented in README.md) will be sufficient.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

