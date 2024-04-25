.. _fi_gaussian_process:

Gaussian Process: light-curve fits
==================================

PISCOLA uses `tinygp <https://tinygp.readthedocs.io/en/stable/>`_ to fit the multi-colour light curves. These are fit in 2D, i.e. `flux` (plus `errors`) as a function of `time` and `wavelength` at the same time. This allows PISCOLA to do a more informative interpolation and extrapolation.

The available kernels are: ``Matern32`` (Mátern-3/2), ``Matern52`` (Mátern-5/2) and ``ExpSquared`` (Squared-Exponential).

For information about the available functions, check the :ref:`Gaussian Process functions <gaussian_process>`.
