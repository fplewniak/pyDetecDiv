Getting started
===============

Requirements
------------
The pyDetecDiv package requires at least Python 3.10

Installation
------------
Easy installation (Linux)
^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install pyDetecDiv on a Linux box is to download the [pydetecdiv-0.1.0-linux64.tar.gz](https://github.com/fplewniak/pyDetecDiv/releases/download/v0.1.0/pydetecdiv-0.1.0-linux64.tar.gz).
Then unpack it in a place of your choice and run the pydetecdiv executable you will find in the pydetecdiv directory.
For a more convenient usage, you may place a link or a copy of the executable in your favourite PATH directory.

Support for easy installation on other operating systems will also be available in the future.

Installation from source distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is recommended to install pyDetecDiv in a conda environment (miniconda) to avoid conflicts.
You will need miniconda anyway to manage and run external tools from within the pyDetecDiv interface.
The minimum compatible Python version is 3.10.

At the moment, pyDetecDiv still requires BioImageIT to be installed but this requirement should not make it into next versions.

Installing in a miniconda environment
"""""""""""""""""""""""""""""""""""""

**Create a conda environment with python:**

.. code-block::

	conda create -n pyDetecDiv python=3.11

Then, you may activate the newly created environment before proceeding with the installation:

.. code-block::

	conda activate pyDetecDiv

**Install BioImageIT:**
BioImageIT needs to be installed manually as there is no pip package available.

.. code-block::

	git clone https://github.com/bioimageit/bioimageit_formats.git
	pip install -e ./bioimageit_formats
	git clone https://github.com/bioimageit/bioimageit_core.git
	pip install -e ./bioimageit_core
	python bioimageit_core/config.py "$USER" "CONDA"

Complete the BioImageIT configuration according to [BioImageIT documentation](https://bioimageit.github.io/bioimageit_core/install.html#configure-bioimageit-core).

**Install pyDetecDiv:**
PyDetecDiv requires the `h5py` package which itself requires `hdf5` libraries which must be installed  before installing the python bindings as it seems that pip cannot install them:

.. code-block::

	conda install hdf5

Then, download the code and install the application:

.. code-block::

	git clone https://github.com/fplewniak/pyDetecDiv.git
	pip install -e ./pyDetecDiv

That's it. All requirements should be automatically installed in your miniconda environment.

**To run the application, you may use:**

.. code-block::

	cd ./pyDetecDiv/src
	python pydetecdiv/app/pydetecdiv_app.py

To avoid the necessity to start the application from the `pyDetecDiv/src` directory, you may want to add it to your `PYTHONPATH` environment variable.

