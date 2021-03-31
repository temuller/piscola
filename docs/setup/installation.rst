
.. _installation:

Installation
========================

PISCOLA can be installed in different way depending on the what the user is looking for. Have in mind that PISCOLA requires `other packages <https://github.com/temuller/piscola/blob/master/requirements.txt>`_ to work.

Using pip
########################

One of the easiest way of installing PISCOLA is by using `pip <https://pip.pypa.io/en/stable/>`_:

.. code::

	pip install piscola

Using conda
########################

Another option is to use `conda <https://docs.conda.io/en/latest/>`_ (**not implemented yet**):

.. code::

	conda install -c conda-forge piscola

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

Using pip + git
########################

Yet another option is to install the code directly from the repository (some sort of "from source") using pip and git:

.. code::

	pip install -e git+https://github.com/temuller/piscola@master#egg=piscola

Use this option only if you know what you are doing.


SFD dust maps
########################

PISCOLA uses dust maps taken from the `sfdmap repository <https://github.com/kbarbary/sfdmap>`_ to calculate extinction. These need to go under the :code:`piscola/sfddata-master` directory (they will already be there if you cloned the repository). **They should automatically be downloaded if they are not found by the code**, but if for any reason they are not, there are a couple of ways to add them:

	1. Manually download the ``fits`` files from one of these links: `option1 <https://github.com/kbarbary/sfddata/>`_ or `option2 <https://github.com/temuller/piscola/tree/master/piscola/sfddata-master>`_

	2. Navigate to the ``piscola/sfddata-master`` directory (you can use ``pip show piscola`` to find where PISCOLA is installed) and run this command on a terminal: ``wget https://github.com/kbarbary/sfddata/archive/master.tar.gz && tar xzf master.tar.gz``


Test the installation
########################

There are different packages to test the installation, but the recommended ones are `pytest <https://docs.pytest.org/en/stable/>`_ and `nose <https://nose.readthedocs.io/en/latest/>`_, specially pytest as it give more details about the tests:

.. code::

	python -m pytest -v tests

The command above would need to be run from the source directory.

