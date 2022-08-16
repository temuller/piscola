
.. _installation:

Installation
========================

PISCOLA can be installed in different way depending on the what the user is looking for. Have in mind that PISCOLA requires `other packages <https://github.com/temuller/piscola/blob/master/requirements.txt>`_ to work. I recommend the use of environments for the installation (see :ref:`envconf`).

Using pip
########################

One of the easiest way of installing PISCOLA and the recommended option is by using `pip <https://pip.pypa.io/en/stable/>`_:

.. code::

	pip install piscola

This option install PISCOLA with all its requirements. You might want to add the ``--user`` flag if you install this on a server, unless you are already in an environment.

From source
########################

To install the code from source, do the following:

.. code::

	git clone https://github.com/temuller/piscola.git
	cd piscola
	pip install .

This will clone the repository locally (`git <https://git-scm.com/>`_ needs to be installed), change directory and install the package. **Note:** Have in mind that installing from source could install an unstable version (PISCOLA is being developed on a separate branch, though).

Using pip + git
########################

Yet another option is to install the code directly from the repository (some sort of "from source") using pip and git:

.. code::

	pip install git+https://github.com/temuller/piscola.git


Test the installation
########################

There are different packages to test the installation, but the recommended one is `pytest <https://docs.pytest.org/en/stable/>`_. To test the installation, one must be in the directory where PISCOLA was installed and run the following command:

.. code::

	pytest -v

