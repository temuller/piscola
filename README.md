# Python for Intelligent Supernova-COsmology Light-Curve Analysis: PISCOLA

**Type Ia Supernova Light-curve fitting code in python**

Dependencies:
  - python 3.6+
  - numpy
  - matplotlib
  - pandas
  - lmfit
  - peakutils
  - george
  - emcee
  - extinction
  - sfdmap
  
It is recommended to create a conda environment:

	`conda config --add channels conda-forge`

	`conda create -n pisco numpy matplotlib pandas lmfit peakutils george emcee extinction sfdmap`

Otherwise, you can install the dependencies by typing `pip install -r dependencies.txt` on your terminal.


In order to use PISCOLA in any directory, you will need to clone the repository and install it by typing `python setup.py install` inside the directory.
