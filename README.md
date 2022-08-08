# galaxy-galaxy-sel
This repository contains programs written for a research project of mine, with the objective of formulating a selection function for [Dark Energy Survey (DES)](https://github.com/DarkEnergySurvey) galaxy-galaxy strong lens candidates—i.e. galaxies which may be strongly lensing a distant background galaxy. We do this through statistical comparison of photometric properties between a population of luminous red galaxies (LRGs) and a sample of galaxy-scale lenses.

The population of LRGs was randomly selected from the DES Y3 redMaGiC catalog ([Pandey, et al., 2021](https://arxiv.org/abs/2105.13545); [Rozo, et al., 2016](https://academic.oup.com/mnras/article/461/2/1431/2608400)), and the sample of galaxy-scale lenses was selected from a catalog of candidates in DES Y3, presented in [Jacobs, et al. \(2019\)](https://iopscience.iop.org/article/10.3847/1538-4365/ab26b6). These candidates were identified using convolutional neural networks, trained on photometric data.

June 2021 - Present

### Software

* NumPy
* SciPy
* matplotlib
* seaborn
* Astropy
* Photutils
* Imfit
* emcee
* SEP (Source Extractor in Python)
