
.. _installation:

Installation
========================

PISCOLA can be installed in different way depending on the what the user is looking for. Have in mind that PISCOLA requires `other packages <https://github.com/temuller/piscola/blob/master/requirements.txt>`_ to work.

Using pip
########################

One of the easiest way of installing PISCOLA is by using pip:

.. code::

	pip install piscola

Using conda
########################

Another option is to use conda (**not implemented yet**):

.. code::

	conda install -c conda-forge piscola

From source
########################

To install the code from source, do the following:

.. code::

	git clone https://github.com/temuller/piscola.git

This will clone the repository locally. You will need to install the requirements first:

.. code::

	pip install -r requirements.txt

And finally proceed to install the code in the usual way:

.. code::

	python setup.py install
