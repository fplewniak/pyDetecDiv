Getting started
===============

Requirements
------------
The pyDetecDiv package requires at least Python 3.10

Installation
------------
Known issue
^^^^^^^^^^^
On Ubuntu distribution with PySide6 version 6.5, you may get the following error:

.. code-block::

	qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
	This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

This may be due to a missing xcb library that can be identified by issuing the following command:

.. code-block::

	ldd <...>/lib/python3.11/site-packages/PySide6/Qt/plugins/platforms/libqxcb.so

Then install the missing library (for example, libxcb-cursor0 on a Ubuntu 22.04 installation):

.. code-block::

	sudo apt install libxcb-cursor0

Easy installation (Linux)
^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install pyDetecDiv on a Linux box is to download the [pydetecdiv-0.3.0-linux64.tar.gz](https://github.com/fplewniak/pyDetecDiv/releases/download/v0.3.0/pydetecdiv-0.3.0-linux64.tar.gz).
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

**Install pyDetecDiv:**

Download the code and install the application:

.. code-block::

	git clone https://github.com/fplewniak/pyDetecDiv.git
	pip install -e ./pyDetecDiv

That's it. All requirements should be automatically installed in your miniconda environment.

**To run the application, you may use:**

.. code-block::

	cd ./pyDetecDiv/src
	python pydetecdiv/app/pydetecdiv_app.py

To avoid the necessity to start the application from the `pyDetecDiv/src` directory, you may want to add it to your `PYTHONPATH` environment variable.

