# Installation

## Linux / MacOS

Conda installation:

```bash
conda install fegrow -c conda-forge
```

We recommend using [mambaforge] which is a flavour of conda. 
Conda struggles with resolving the environment.

To use the latest git version, download the code 
and use the provided environment file *environment.yml*

```bash
git clone https://github.com/cole-group/FEgrow.git
cd FEgrow
mamba env create -f environment.yml
conda activate fegrow
pip install --no-deps . # the repository directory
```

In order to ensure that the environment is available 
in your jupyter, you can then use

```bash
python -m ipykernel install --user --name=fegrow
```

!!! Warning
    Gnina stage is not supported on MacOS because CUDA's 
    is not available on MacOS 
    ([see issue][gnina])


## Windows

We have not verified whether this toolkit works on Windows
and we expect issues with the dependancy **openmm-torch**.


[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[gnina]: https://github.com/gnina/gnina/issues/129