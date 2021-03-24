.. _CondaConfigurations:

Environment configuration
=============================

It is usually a good idea to install some packages on a separate environment so you don't mess up with your main Python environment. The preferred option to setup your environment is through conda environment as follows:

.. code::

	conda create --name <env_name> --file requirements.txt

This will create the environment with the required packages for PISCOLA to work. 

**Note:** the versions of the packages install with conda are not always the same as the ones installed with pip.
