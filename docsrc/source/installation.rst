Installation
============

Linux / MacOS
-------------

Conda installation:

.. code-block:: Bash

    mamba install fegrow -c conda-forge

To use the latest git version, download the code and use the provided
environment file *env.yml*. We recommend using Mambaforge (https://github.com/conda-forge/miniforge#mambaforge)
which is a flavour of conda. Conda struggles with resolving the environment. 

.. code-block:: Bash

    git clone https://github.com/cole-group/FEgrow.git
    cd FEgrow
    mamba env create -f env.yml
    conda activate fegrow
    pip install . # the repository directory

In order to ensure that the environment is available in your jupyter, you can then use::

    python -m ipykernel install --user --name=fegrow

..

    WARNING: Gnina stage is not supported on MacOS because of CUDA's incompatibility with MacOS's GPUs (`the relevant issue <https://github.com/gnina/gnina/issues/129>`_)

Windows
-------

We have not verified whether this toolkit works on Windows
and we expect issues with the dependancy **openmm-torch**.
