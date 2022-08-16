.. _envconf:

Environment configuration
=============================

It is usually a good idea to install some packages on a separate environment so you don't mess up with your main Python environment. You have two main options: `conda <https://docs.conda.io/en/latest/>`_ and `pyenv <https://github.com/pyenv/pyenv>`_.


Conda environment
########################

To setup your environment through conda environment use:

.. code::

	conda create --name <env_name> pip

By installing pip in the environment, you can then use it without affecting your packages outside the environment. By adding the ``--file requirements.txt`` flag you will create the environment with the required dependencies for PISCOLA to work (the ``requirements.txt`` file can be downloaded from the repository). Have in mind that this will use the conda versions of the packages.

**Note:** the versions of the packages installed with conda are not always the same as the ones installed with pip.


Pyenv + virtualenv
########################

If your prefer to use pyenv, you can use:

.. code::

	pyenv virtualenv <env_name>

