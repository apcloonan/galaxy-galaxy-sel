## galaxy-galaxy-sel
This repository contains programs written for a research project of mine, with the objective of formulating a selection function for [Dark Energy Survey (DES)](https://github.com/DarkEnergySurvey) galaxy-galaxy strong lens candidates—i.e. galaxies which may be strongly lensing a distant background galaxy. We do this through statistical comparison of photometric properties between a population of luminous red galaxies (LRGs) and a sample of galaxy-scale lenses.

We randomly select a population of 10,000 LRGs from the DES Y3 redMaGiC catalog ([Pandey et al. 2021](https://arxiv.org/abs/2105.13545); [Rozo et al. 2016](https://academic.oup.com/mnras/article/461/2/1431/2608400)), and we select a sample of galaxy-scale lenses from a catalog of candidates presented in [Jacobs et al. \(2019\)](https://iopscience.iop.org/article/10.3847/1538-4365/ab26b6). These candidates were identified in the DES Y3 imaging catalogs using convolutional neural networks, trained on both observational and simulated photometric data. Spectroscopic confirmation for a handful of these candidates is presented in [Tran et al. 2022](https://arxiv.org/abs/2205.05307).

June 2021 - Present

### Software

* NumPy
* SciPy
* matplotlib
* seaborn
* pandas
* Astropy
* Photutils
* Imfit
* emcee
* SEP (Source Extractor in Python)
