---
title: 'PISCOLA: Python for Intelligent Supernova-COsmology Light-curve Analysis'
tags:
  - Python
  - astronomy
  - supernova
  - cosmology

authors:
  - name: Tomás E. Müller Bravo^[t.e.muller-bravo@soton.ac.uk]
    orcid: 0000-0003-3939-7167
    affiliation: 1
  - name: Mark Sullivan
    affiliation: 1
  - name: Mat Smith
    affiliation: "1, 2" # (Multiple affiliations must be quoted)

affiliations:
 - name: Department of Physics and Astronomy, University of Southampton, Southampton, Hampshire, SO17 1BJ, UK
   index: 1
 - name: Université de Lyon, Université de Lyon 1, Villeurbanne; CNRS/IN2P3, Institut de Physique des Deux Infinis, F-69622 Lyon, France
   index: 2

date: March 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# mnras-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
---

# Summary

Type Ia Supernovae (SNe Ia) are stellar explosions that have been studied for many years as standardisable candles 
for cosmological distance measurement. Their multi-colour light curves need to be fitted and corrected (e.g., 
extinction and $K$-correction) in order to be standardised. Several light-curve fitters exist now a day for this.
With future surveys, such as LSST, the sample of these objects will rapidly increase by orders of magnitude, reducing the 
statistical uncertainties. However, a large portion of the error budget is dominated by systematic uncertatinties.


# Statement of need

`PISCOLA` is a new light-curve fitting code developed in Python. This package is user-friendly and well-documented. One 
of the main goals behind this code is to allow the users to have access and undertand the different steps of the light-curve 
fitting and correction process, so the community can contribute to its improvement. `PISCOLA` relies on gaussian process [@gp], 
a data-driven bayesian method, to fit the light curves in 2D (luminosity as a function of time and wavelength), using the
package `george` `[@george]` for it. The *mangling* function, used in the $K$-correction, is also calculated by using gaussian 
process. Finally, the standard light-curve parameters ($\m_B^{max}$, $\Delta$m$_{15}(15)$ and $(B-V)_{max}$) can be estiamted.

Several light-curve fitters (e.g., SALT2 [@salt2], SiFTO [@sifto], SNooPy [@snoopy] have proven to be great tools for 
supernova cosmology, however, most of them present some disadvantage such as: their limitation to working with optical data only 
(SNe Ia are better suited for cosmology in the near-infrared; e.g., [@Elias81], [@Freedman09]), except for SNooPy, and their 
susceptibility to biases [@Kessler09b], as they rely on templates for the fits. PISCOLA, being a data-driven tool, does not 
suffer from this disadvantages. For this reason, this code has the potential to produce better results and possibly help understand 
different systematic biases the other codes suffer from.


# Acknowledgements

TMB would like to thank the LSSTC-DSFP which allowed him to develop many computational skills and tools for the development of PISCOLA.


# References
