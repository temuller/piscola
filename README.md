# Python for Intelligent Supernova-Cosmology Light-Curve Analysis: PISCoLA

**Type Ia Supernova Light-curve fitting code in python**

Dependencies:
  - Python 3.6+
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

`conda create -n pisco matplotlib pandas lmfit peakutils george emcee extinction multiprocess`

but `sfdmap` would need to be installed manually with `pip`.

Otherwise, you can install the dependencies by typing `pip install -r dependencies.txt` on your terminal.


In order to use PISCoLA in any directory, you will need to install it: `python setup.py install`.
