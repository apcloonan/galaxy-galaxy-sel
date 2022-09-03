
### Pipeline for Final Photometric Modeling of Galaxy Light Profiles

### Aidan Cloonan
### Last Updated September 2022

### I'm starting with the lensing galaxies in order to generate a population distribution in R_eff,r - z_phot space

# -------------------------------

# math, array manipulation, etc.
import numpy as np

import os
import sys

# timing
from timeit import default_timer
from tqdm import tqdm

import pandas as pd

import astropy.io.fits as fits
from astropy.table import Table                    # Table data structure
import astropy.units as u

# necessary utilities from scipy, astropy and photutils
from scipy.optimize import differential_evolution
from scipy.ndimage import maximum_filter, gaussian_filter
from astropy.modeling import functional_models
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, SigmaClip
from photutils.background import Background2D, StdBackgroundRMS
from photutils.segmentation import deblend_sources, SegmentationImage, detect_sources

# plots
import matplotlib.pyplot as plt

#%matplotlib inline

# MCMC sampling package
import emcee

# SEP
import sep

sys.path.append('/Users/aidan/Desktop/sl_project/galaxy-galaxy-sel/')

from lensphoto import LensPhoto, fwhm2sigma

# -------------------------------

# the warnings that have shown up don't appear to cause problems 

import warnings
warnings.filterwarnings("ignore")

# -------------------------------

# program inputs: FITS files storing images, the location and filename to save results to

path = '/Users/aidan/Desktop/sl_project/production/'

hdulist_r = fits.open(path + 'lens_images_r.fits')
hdulist_g = fits.open(path + 'lens_images_g.fits')

results_filename = path + 'lens_results.csv'

fwhm_r = 0.95
fwhm_g = 1.11

std_psf_r = fwhm2sigma(fwhm_r)
std_psf_g = fwhm2sigma(fwhm_g)

# -------------------------------

# to help keep the incoming mess of lists slightly less cluttered

def add_results_onto_columns(sample_arr, l16, l50, l84):
    
    l16.append(np.percentile(sample_arr, 16))
    l50.append(np.percentile(sample_arr, 50))
    l84.append(np.percentile(sample_arr, 84))
    
    return l16, l50, l84

# -------------------------------

# initializations

desids = []
photozs = []

pa16 = []
pa50 = []
pa84 = []
e16 = []
e50 = []
e84 = []
rr16 = []
rr50 = []
rr84 = []
nr16 = []
nr50 = []
nr84 = []
rg16 = []
rg50 = []
rg84 = []
ng16 = []
ng50 = []
ng84 = []

rmag16 = []
rmag50 = []
rmag84 = []
gmag16 = []
gmag50 = []
gmag84 = []
gr16 = []
gr50 = []
gr84 = []

# -------------------------------

# begin

for ind in np.arange(len(hdulist_r)):

    img_r = hdulist_r[ind].data
    img_g = hdulist_g[ind].data

    # find coordinate-based galaxy name and photometric redshift, append to respective lists

    desid = hdulist_r[ind].header['cand']
    photoz = hdulist_r[ind].header['gphoto_z']
    
    assert (desid == hdulist_g[ind].header['cand'])
    
    print('--------------------------------------------\n\nNow modeling galaxy {} (index {})\n\n--------------------------------------------'.format(desid, ind))

    desids.append(desid)
    photozs.append(photoz)

    # run model

    photo = LensPhoto(img_r = img_r, img_g = img_g
                      , std_psf_r = std_psf_r, std_psf_g = std_psf_g
                      , desid = desid
                     )

    run, dist = photo.mcmc_run()
    samples = photo.dist.reshape(-1, photo.ndim)

    rmag_arr, gmag_arr = photo.mag_arrays()

    pa16, pa50, pa84 = add_results_onto_columns(samples[:,2], pa16, pa50, pa84)
    e16, e50, e84 = add_results_onto_columns(samples[:,3], e16, e50, e84)
    nr16, nr50, nr84 = add_results_onto_columns(samples[:,4], nr16, nr50, nr84)
    rr16, rr50, rr84 = add_results_onto_columns(samples[:,5], rr16, rr50, rr84)
    ng16, ng50, ng84 = add_results_onto_columns(samples[:,6], ng16, ng50, ng84)
    rg16, rg50, rg84 = add_results_onto_columns(samples[:,7], rg16, rg50, rg84)

    rmag16, rmag50, rmag84 = add_results_onto_columns(photo.rmag_arr, rmag16, rmag50, rmag84)
    gmag16, gmag50, gmag84 = add_results_onto_columns(photo.gmag_arr, gmag16, gmag50, gmag84)
    gr16, gr50, gr84 = add_results_onto_columns(photo.gr_arr, gr16, gr50, gr84)
    
    print('\nBest fit results (medians) for {}:\n'.format(desid))
    print('r-band mag = {}'.format(rmag50[-1]))
    print('g-band mag = {}'.format(gmag50[-1]))
    print('g-r color = {}'.format(gr50[-1]))
    print('n_r = {}'.format(nr50[-1]))
    print('R_eff,r = {}'.format(rr50[-1]))
    print('n_g = {}'.format(ng50[-1]))
    print('R_eff,g = {}'.format(rg50[-1]))
    print('ellipticity = {}\n'.format(e50[-1]))

# -------------------------------

# construct resulting dataframe and save it

data = {'cand_name': desids,
        'photo_z': photozs,
        'rmag_16': rmag16,               # 1
        'rmag_50': rmag50,
        'rmag_84': rmag84,
        'gmag_16': gmag16,               # 2
        'gmag_50': gmag50,
        'gmag_84': gmag84,
        'gr_16': gr16,               # 3
        'gr_50': gr50,
        'gr_84': gr84,
        'halflightrad_r_16': rr16,               # 4
        'halflightrad_r_50': rr50,
        'halflightrad_r_84': rr84,
        'sersicindex_r_16': nr16,               # 5
        'sersicindex_r_50': nr50,
        'sersicindex_r_84': nr84,
        'halflightrad_g_16': rg16,               # 6
        'halflightrad_g_50': rg50,
        'halflightrad_g_84': rg84,
        'sersicindex_g_16': ng16,               # 7
        'sersicindex_g_50': ng50,
        'sersicindex_g_84': ng84,
        'ellipticity_16': e16,               # 8
        'ellipticity_50': e50,
        'ellipticity_84': e84,
        'positionangle_16': pa16,
        'positionangle_50': pa50,
        'positionangle_84': pa84
       }

results = pd.DataFrame(data)

results.to_csv(results_filename)

