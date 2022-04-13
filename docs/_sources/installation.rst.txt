Installation
============

Linux / MacOS
-------------

For conda installation, please use the provided environment file *env.yml*:

.. code-block:: Bash

    conda env create -f env.yml
    conda activate fegrow

Alternatively, you can install the dependancies manually in the env.yml and use pip to install the package:

.. code-block:: Bash

    cd fegrow
    pip install .

In order to ensure that the environment is available in your jupyter, you can use::

    python -m ipykernel install --user --name=fegrow

..

    WARNING: Gnina stage is not supported on MacOS because of CUDA's incompatibility with MacOS's GPUs (`the relevant issue <https://github.com/gnina/gnina/issues/129>`_)

Windows
-------

We have not verified whether this toolkit works on Windows
and we expect issues with the dependancy **openmm-torch**.
