
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

	pip install -e git+git://github.com/temuller/piscola@master#egg=piscola

Use this option only if you know what you are doing.


Test the installation
########################

There are different packages to test the installation, but the recommended ones are `pytest <https://docs.pytest.org/en/stable/>`_ and `nose <https://nose.readthedocs.io/en/latest/>`_, specially pytest as it give more details about the tests:

.. code::

	python -m pytest -v tests

