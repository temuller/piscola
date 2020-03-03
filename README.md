# Python for Intelligent Supernova-Cosmology Light-Curve Analysis: PISCoLA

**Type Ia Supernova Light-curve fitting code in python**

Dependencies:
  - Python 3.6 (or greater)
  - matplotlib 3.0.3
  - pandas 0.24.2
  - lmfit 0.9.12
  - peakutils 1.3.2
  - george 0.3.1
  - extinction 0.4.0
  - sfdmap 0.1.1
  
It is recommended to create a conda environment:

`conda config --add channels conda-forge`

`conda create -n pisco matplotlib pandas lmfit peakutils george extinction multiprocess`

but `sfdmap` would need to be installed manually with `pip`.
