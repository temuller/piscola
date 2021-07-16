
.. _installation:

Installation
========================

PISCOLA can be installed in different way depending on the what the user is looking for. Have in mind that PISCOLA requires `other packages <https://github.com/temuller/piscola/blob/master/requirements.txt>`_ to work. I recommend the use of environments for the installation (see :ref:`envconf`). The SFD maps for the calculation of Milky Way dust extinction need to be downloaded separately (see the `SFD dust maps`_ section below).

Using pip
########################

One of the easiest way of installing PISCOLA and the recommended option is by using `pip <https://pip.pypa.io/en/stable/>`_:

.. code::

	pip install piscola

You might want to add the ``--user`` flag if you install this on a server, unless you are already in an environment.

Using conda
########################

Another option is to use `conda <https://docs.conda.io/en/latest/>`_:

.. code::

	conda install -c temuller piscola

This might not always download the latest version (check the repository for the versions available).

From source
########################

To install the code from source, do the following:

.. code::

	git clone https://github.com/temuller/piscola.git

This will clone the repository locally (`git <https://git-scm.com/>`_ needs to be installed). You will need to change directory (:code:`cd piscola`) and install the dependencies:

.. code::

	pip install -r requirements.txt

And finally proceed to install the code in the usual way:

.. code::

	python setup.py install

If you use the ``--user`` flag, the package will be installed in the user ``site-packages``. Alternatively, you can use the ``--home`` or ``--prefix`` option to install the package in a different location (where you have the necessary permissions). The last command also installs the dependencies, but some of them are needed for the installation of the package (like ``numpy``).

Using pip + git
########################

Yet another option is to install the code directly from the repository (some sort of "from source") using pip and git:

.. code::

	pip install git+https://github.com/temuller/piscola.git

.. _SFD dust maps:

SFD dust maps
########################

PISCOLA uses dust maps taken from the `sfdmap repository <https://github.com/kbarbary/sfdmap>`_ to calculate extinction. By default, PISCOLA looks under the :code:`src/piscola/sfddata-master` directory (they will already be there if you cloned the repository), although you can change the path where the dust maps are found (see the `Extinction correction <extinction_correction>`_ section). There are a few ways to download these dust maps:

	1. Manually download the files from this links: `https://github.com/kbarbary/sfddata <https://github.com/kbarbary/sfddata/>`_.

	2. Run this command on a terminal: ``wget https://github.com/kbarbary/sfddata/archive/master.tar.gz && tar xzf master.tar.gz``.

	3. Use the ``download_dustmaps.py`` script included in the repository which will download and extract the files in a directory with the name ``sfddata-master``.

For the last option you can run this from a terminal (this requires ``wget`` which is in the package's requirements):

.. code::

	chmod -x download_dustmaps.py
	python download_dustmaps.py <path/where/to/download>

If the argument is empty, the ``sfddata-master`` directory with SFD maps files will be downloaded in your current directory, or you can use ``piscola`` as argument, which will download the files under the default directory (:code:`src/piscola/sfddata-master`) if PISCOLA is installed. You can also give a different path as argument if desired.


Test the installation
########################

There are different packages to test the installation, but the recommended ones are `pytest <https://docs.pytest.org/en/stable/>`_ and `nose <https://nose.readthedocs.io/en/latest/>`_, specially pytest as it give more details about the tests. To test the installation I recomment cloning the repository, as it includes all the test files, and run the following command in the repository directory:

.. code::

	python -m pytest -v tests

