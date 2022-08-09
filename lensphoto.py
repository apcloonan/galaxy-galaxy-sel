
### Photometric Analysis Toolkit for Galaxy-Galaxy Strong Lenses

### Aidan Cloonan
### Last Updated August 2022

# -------------------------------

### Next steps:

#   (DONE) implement magnitude (i.e. MAG_AUTO) measurements using SEP for every posterior PDF evaluation
#             perform measurements on generated model images
#             see this link for info on how to do this: https://sep.readthedocs.io/en/v1.1.x/apertures.html#equivalent-of-flux-auto-e-g-mag-auto-in-source-extractor

#   in the object detection and masking algorithm, add an r-band component to the mask and then test

#   set up a production run in a notebook which does MCMC modeling for a handful of systems
#   find a way to compile the photometric redshift in the process
#   sigma clip outliers in magnitudes just in case (they definitely will show up in g-band at least)

# -------------------------------

### IMPORTS

import os
import sys

# math, array manipulation, etc.
import numpy as np

# timing
from timeit import default_timer
from tqdm import tqdm

# data structures from astropy
import astropy.io.fits as fits
from astropy.table import Table

# astronomical image fitting software Imfit, wrapped in Python
# https://github.com/perwin/pyimfit
# https://pyimfit.readthedocs.io/en/latest/
import pyimfit

# MCMC sampling package
import emcee

# necessary utilities from scipy, astropy, and photutils
from scipy.ndimage import maximum_filter
from scipy.stats import multivariate_normal
from astropy.stats import SigmaClip, gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
from photutils.background import Background2D, StdBackgroundRMS
from photutils.segmentation import detect_sources, SegmentationImage, deblend_sources

# plots
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# SExtractor in Python
import sep

# -------------------------------

def fwhm2sigma(fwhm, arcsec_pix = 0.263   # arcsec per pixel, it's 0.263 for DECam
              ):
    '''
    convert FWHM to a standard deviation for a Gaussian distribution
    
    This is meant for Gaussian light distributions (in our case, a rough PSF), so
    the units are in arcseconds
    
    In the DES DR2 imaging catalogs, median PSF FWHMs are:
        r-band --> 0.95 arcsec
        g-band --> 1.11 arcsec
    '''
    fwhm_pix = fwhm / arcsec_pix
    std = fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
    
    return std

class LensPhoto:

    def __init__(self, img_r, img_g, desid, std_psf_r, std_psf_g
                 , box_size = 10, dilation_factor=3, threshold_mult = 0.5, source_npix = 25       # masking parameters
                 , ndim=8, nwalkers=20, steps=600, burnin = 300
                ):
    
        self.img_r = img_r
        self.img_g = img_g
        
        self.desid = desid
        
        self.std_psf_r = std_psf_r
        self.std_psf_g = std_psf_g
        
        self.nrow, self.ncol = self.img_r.shape
        
        self.params = ['x_{0}', 'y_{0}', '\\theta', 'E'
                       , 'n_{r}', 'R_{1/2, r}'
                       , 'n_{g}', 'R_{1/2, g}']
        
        # bounds on posterior PDF sampling
        self.Ie_bounds = [1e-3, 300];
        self.r_bounds = [0.1, 30];
        self.n_bounds = [0.5, 8];
        self.x_bounds = [self.ncol/2 - 3, self.ncol/2 + 3];
        self.y_bounds = [self.nrow/2 - 3, self.nrow/2 + 3];
        self.e_bounds = [0, 1];
        self.th_bounds = [0, 180]
        
        self.Iemin, self.Iemax = self.Ie_bounds.copy();
        self.rmin, self.rmax = self.r_bounds.copy();
        self.nmin, self.nmax = self.n_bounds.copy();
        self.x0min, self.x0max = self.x_bounds.copy();
        self.y0min, self.y0max = self.y_bounds.copy();
        self.emin, self.emax = self.e_bounds.copy();
        self.thmin, self.thmax = self.th_bounds.copy()
        
        self.psf_r, self.psf_g = self.construct_psfs()
        
        img_gal_r, img_bg_r, stds_r, std_bg_r = self.calc_err_and_gal(self.img_r)
        self.img_gal_r = img_gal_r; 
        self.img_bg_r = img_bg_r; 
        self.stds_r = stds_r; 
        self.std_bg_r = std_bg_r
        
        img_gal_g, img_bg_g, stds_g, std_bg_g = self.calc_err_and_gal(self.img_g)
        self.img_gal_g = img_gal_g;
        self.img_bg_g = img_bg_g; 
        self.stds_g = stds_g; 
        self.std_bg_g = std_bg_g
        
        masked_img_gal_r, masked_img_gal_g = self.create_mask(box_size = box_size, dilation_factor=dilation_factor
                                                              , threshold_mult = threshold_mult, source_npix = source_npix
                                                             )
        
        self.img_gal_r = masked_img_gal_r
        self.img_gal_g = masked_img_gal_g
        
        self.ndim = ndim
        self.nwalkers = nwalkers
        self.steps = steps
        self.burnin = burnin
        
        if self.steps <= self.burnin:
            raise NotEnoughStepsError('There should be more sampling steps than burn-in steps.')

    def construct_psfs(self):
        '''
        take the given full widths at half maximum in r&g bands and
        creates a pair of model PSF images
        
        these go on to be convolved with the model brightness profiles
        '''
        
        x, y = np.meshgrid(np.arange(self.ncol), np.arange(self.nrow))
        pos = np.dstack((x, y))
        
        # r band
        
        psf_dist_r = multivariate_normal((np.array([self.nrow, self.ncol])-1)/2
                                               , np.full(2, self.std_psf_r))

        psf_r = psf_dist_r.pdf(pos)
        
        # g band
        
        psf_dist_g = multivariate_normal((np.array([self.nrow, self.ncol])-1)/2
                                               , np.full(2, self.std_psf_g))

        psf_g = psf_dist_g.pdf(pos)
        
        return psf_r, psf_g 
    
    def calc_err_and_gal(self, img_i, num=1.0):
        '''

        calculates an intensity uncertainty map corresponding to the input image
    
        also subtracts background and returns the grid of I_gal values, which is what the Sérsic model is fit to
    
        inputs:
        --------
    
        img        ---    2D array; a DES redMaGiC image with an elliptical galaxy that a Sérsic
                          model is being fit to  
    
        outputs:
        --------
    
        img_gal    ---    2D array; alternatively img_gal_copy (explained below); the grid of I_gal values
                          calculated from estimation of a constant I_bg
    
        std_arr    ---    2D array; grid of uncertainties in corresponding pixel values from the DES image
  
        '''
    
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    
        bg_pixels = sigma_clip(img_i, masked=False, axis=None)
    
        ## I_bg and std_bg
    
        img_bg = np.median(bg_pixels)  # I_bg
    
        bkgrms = StdBackgroundRMS(sigma_clip)
        std_bg = bkgrms(img_i) 
        
        #std_bg = np.std(bg_pixels)
    
        ## I_gal
    
        img_gal = img_i - img_bg          # I_tot = I_gal + I_bg
    
        std_gal = np.sqrt(num * img_gal) 
    
        invalid_sgal_ind = (std_gal == np.inf) | (std_gal == -np.inf)
        std_gal[invalid_sgal_ind] = 0
    
        std_gal = np.nan_to_num(std_gal)
    
        ## estimate uncertainties

        std_arr = np.sqrt(std_gal**2 + std_bg**2)
        #std_arr = np.full_like(img_i, test_bg)
        print(std_arr)

        return img_gal, img_bg, std_arr, std_bg
    
    def create_mask(self, box_size=10, dilation_factor=3, threshold_mult = 0.5, source_npix = 25):
        '''
        Takes a raw input fits image and constructs a mask to filter out light 
        from other sources apart from the lens/galaxy. Returns the masked image,
        where pixel values from other sources are set to 0.
    
        '''
    
        # construct a background image w/ background noise
        bkg = Background2D(self.img_g, box_size=box_size)
        bkg_img = bkg.background
    
        # calculate RMS of each pixel, used to calculate threshold for source identification
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        bkgrms = StdBackgroundRMS(sigma_clip)
        bkgrms_img = bkgrms(self.img_g) 

        # map of thresholds over which sources are detected
        threshold = bkg_img + (threshold_mult * bkgrms_img)  
    
        # source detection
        sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3).normalize()
        segm = detect_sources(self.img_g, threshold, source_npix, kernel)
    
        # deblending sources, looking for saddles between peaks in flux
        segm_deblend = deblend_sources(self.img_g, segm, npixels=5,
                                       nlevels=32, contrast=0.001)
                
        label = segm_deblend.data[(segm_deblend.data.shape[0]//2, segm_deblend.data.shape[1]//2)]
        other_inds = np.delete(np.arange(1, segm_deblend.nlabels+1), label-1)
    
        deblend_copy = segm_deblend.data.copy()
        source = (deblend_copy == label)
    
        # get pixels from all other sources
        deblend_copy2 = segm_deblend.copy()
        deblend_copy2.keep_labels(other_inds)

        segm_dilated_arr = maximum_filter(deblend_copy2.data, dilation_factor)
        segm_dilated_arr[deblend_copy2.data != 0] = deblend_copy2.data[deblend_copy2.data != 0]
        
        # label central source, which is the lens/galaxy
        segm_dilated_arr[source] = 10000
    
        segm_dilated = SegmentationImage(segm_dilated_arr)
    
        other_inds = np.delete(segm_dilated.labels, -1)
        
        # get pixels from all other sources
        segm_dilated.keep_labels(other_inds)
        
        self.segm = segm_dilated.data
    
        self.mask = (self.segm > 0)
        self.unmask = np.invert(self.mask)

        # make values of those pixels 0 in img_gal
        gal_copy_g = self.img_gal_g.copy()
        gal_copy_g[self.mask] = 0
        gal_copy_r = self.img_gal_r.copy()
        gal_copy_r[self.mask] = 0
    
        return gal_copy_r, gal_copy_g
    
    def quick_fit(self):
        '''
        makes a quick fit using imfit, used to find initial vector for MCMC
        '''
        
        ### r-band
        
        self.model_desc_r = pyimfit.SimpleModelDescription()
        
        # define the limits on the central-coordinate X0 and Y0 as +/-10 pixels relative to initial values
        # (note that Imfit treats image coordinates using the IRAF/Fortran numbering scheme: the lower-left
        # pixel in the image has coordinates (x,y) = (1,1))
        self.model_desc_r.x0.setValue(self.nrow/2, self.x_bounds)
        self.model_desc_r.y0.setValue(self.ncol/2, self.y_bounds)

        # create a Sersic image function, then define the parameter initial values and limits
        lrg_r = pyimfit.make_imfit_function("Sersic", label="LRG_r")

        lrg_r.I_e.setValue(15, self.Ie_bounds)
        #lrg_r.r_e.setValue(self.Re_r, fixed=True)
        lrg_r.r_e.setValue(5, self.r_bounds)
        lrg_r.n.setValue(3, self.n_bounds)

        lrg_r.PA.setValue(40, self.th_bounds)
        lrg_r.ell.setValue(0.5, self.e_bounds)
        
        self.model_desc_r.addFunction(lrg_r)
        
        self.imfit_r = pyimfit.Imfit(self.model_desc_r
                                     , psf=self.psf_r)
        
        self.imfit_r.loadData(self.img_gal_r
                              , error=self.stds_r
                              , error_type="sigma"
                              , mask=self.segm
                             )

        results_r = self.imfit_r.doFit(getSummary=True)
        bestfit_r = results_r.params
        errs_r = results_r.paramErrs
        
        ### g-band
        
        self.model_desc_g = pyimfit.SimpleModelDescription()
        
        # define the limits on the central-coordinate X0 and Y0 as +/-10 pixels relative to initial values
        # (note that Imfit treats image coordinates using the IRAF/Fortran numbering scheme: the lower-left
        # pixel in the image has coordinates (x,y) = (1,1))
        self.model_desc_g.x0.setValue(self.nrow/2, self.x_bounds)
        self.model_desc_g.y0.setValue(self.ncol/2, self.y_bounds)

        # create a Sersic image function, then define the parameter initial values and limits
        lrg_g = pyimfit.make_imfit_function("Sersic", label="LRG_g")

        lrg_g.I_e.setValue(10, self.Ie_bounds)
        #lrg_g.r_e.setValue(self.Re_g, fixed=True)
        lrg_g.r_e.setValue(5, self.r_bounds)
        lrg_g.n.setValue(1, self.n_bounds)

        lrg_g.PA.setValue(40, self.th_bounds)
        lrg_g.ell.setValue(0.5, self.e_bounds)
        
        self.model_desc_g.addFunction(lrg_g)
        
        self.imfit_g = pyimfit.Imfit(self.model_desc_g
                                     , self.psf_g
                                    )
        
        self.imfit_g.loadData(self.img_gal_g
                              , error=self.stds_g
                              , error_type="sigma"
                              , mask=self.segm
                             )

        results_g = self.imfit_g.doFit(getSummary=True)
        bestfit_g = results_g.params
        errs_g = results_g.paramErrs
        
        #print('r band:\n', results_r)
        #print('\ng band:\n', results_g)
    
        return bestfit_r, bestfit_g#, errs_r, errs_g
        
    def log_priors(self, v):
        '''
        all priors are taken to be uniform distributions
        '''
    
        x0, y0, th, e, nr, rr, ng, rg = v
        
        if ((self.rmin <= rr <= self.rmax) and (self.nmin <= nr <= self.nmax)                   # r-band
            and (self.rmin <= rg <= self.rmax) and (self.nmin <= ng <= self.nmax)               # g-band
            and (self.x0min <= x0 <= self.x0max) and (self.y0min <= y0 <= self.y0max) 
            and (self.emin <= e <= self.emax) and (self.thmin <= th <= self.thmax)):
        
            lnprior = 0
            
        else:
        
            lnprior = -np.inf
            
        return lnprior
    
    def norm_to_notnorm(self, v):
        '''
        takes a sample from the posterior and calculates non-normalized amplitude
        '''
        
        x0, y0, th, e, nr, rr, ng, rg = v
        
        gal_filt_r = self.img_gal_r[self.unmask]
        gal_filt_g = self.img_gal_g[self.unmask]
        
        stds_filt_r = self.stds_r[self.unmask]
        stds_filt_g = self.stds_g[self.unmask]
        
        # r-band
        norm_params_r = np.array([x0, y0, th, e, nr, 1.0, rr])
        norm_profile_r = self.imfit_r.getModelImage(newParameters=norm_params_r)
        norm_filt_r = norm_profile_r[self.unmask]
        
        # g-band
        norm_params_g = np.array([x0, y0, th, e, ng, 1.0, rg])
        norm_profile_g = self.imfit_g.getModelImage(newParameters=norm_params_g)
        norm_filt_g = norm_profile_g[self.unmask]
    
        # find amplitudes, then replace in parameter arrays
        # r-band
        Ie_r = np.sum((norm_filt_r * (gal_filt_r))
                      / (stds_filt_r)**2) / np.sum((norm_filt_r)**2 
                                                   / (stds_filt_r)**2)
        
        # g-band
        Ie_g = np.sum((norm_filt_g * (gal_filt_g))
                      / (stds_filt_g)**2) / np.sum((norm_filt_g)**2 
                                                   / (stds_filt_g)**2)
        
        return Ie_r, Ie_g
        
    def log_posterior(self, v):
    
        ### find values in prior and likelihood pdfs
        
        ## prior
        
        logprior = self.log_priors(v)
        
        ## likelihood
        
        self.x0, self.y0, self.th, self.e, self.nr, self.rr, self.ng, self.rg = v
    
        # find amplitudes, then create parameter arrays
        Ie_r, Ie_g = self.norm_to_notnorm(v)
       
        new_params_r = np.array([self.x0, self.y0, self.th, self.e, self.nr, Ie_r, self.rr])
        new_params_g = np.array([self.x0, self.y0, self.th, self.e, self.ng, Ie_g, self.rg])
        
        # calculate likelihood
        loglike_r = -0.5 * self.imfit_r.computeFitStatistic(new_params_r)
        loglike_g = -0.5 * self.imfit_g.computeFitStatistic(new_params_g)
    
        ## posterior distribution value
        
        logpost = logprior + loglike_r + loglike_g
    
        if np.isnan(logpost):
            logpost = -np.inf
            
        ## magnitude calculations
    
        return logpost
    
    def mcmc_run(self):
        
        fit_r, fit_g = self.quick_fit()
        
        init_fit = np.concatenate((fit_r[:-2], fit_r[-1:], [fit_g[-3], fit_g[-1]]))
        
        # initial vector
        # use np.random.normal instead, and center around min_v, small std
        size = self.ndim * self.nwalkers
        pr = np.random.normal(loc=0, scale=0.01, size=size).reshape((self.nwalkers, self.ndim)) + init_fit
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_posterior)

        sampler.run_mcmc(pr, self.steps, progress=True)
        
        chain = sampler.get_chain()
            
        self.chain = chain
        
        self.dist = self.chain[self.burnin:]
        
        return self.chain, self.dist
    
    def plot_trace(self, cutoff = 750, figsize=(5, 1)):
        
        try:
            nsteps, ndims = np.shape(self.chain)[0], np.shape(self.chain)[2]
        except:
            nsteps, ndims = np.shape(self.chain)[0], 1
        
        medians = np.median(self.chain, axis=1)
    
        sig_lb = np.percentile(self.chain, 16, axis=1)
        sig_ub = np.percentile(self.chain, 84, axis=1)
        
        for d in range(ndims):
        
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
    
            ax.set_ylabel(r'${}$'.format(self.params[d])) # label axis 
            ax.set_xlabel(r'step number')
            
            if ndims == 1:
            
                avr = np.mean(medians[-100:])
            
                xl = np.linspace(-50, len(medians)+50, 50)  # average
                yl = 0 * xl + avr
            
                xc = 0 * xl + cutoff
    
                ax.plot(xl, yl, '--', c='r', lw=1.0)
                ax.plot(xc, xl, '--', c='g', lw=1.0)

                ax.set_xlim(-25, len(medians) + 25)
                ax.set_ylim(np.min(medians) * 0.85, np.max(medians) * 1.15)
            
        
                ax.plot(np.arange(0,nsteps,1), medians, alpha=0.75, lw=0.5, c='darkslateblue', label='d={:d}'.format(d))
                ax.fill_between(np.arange(0,nsteps,1), sig_lb, sig_ub, alpha=0.2)
        
            else:
            
                avr = np.mean(medians[:,d][-100:])
                
                xl = np.linspace(-50, len(medians[:,d])+50, 50)
                yl = 0 * xl + avr
            
                xc = 0 * xl + cutoff
    
                ax.plot(xl, yl, '--', c='r', lw=1.0)
                ax.plot(xc, xl, '--', c='g', lw=1.0)

                ax.set_xlim(-25, len(medians[:,d]) + 25)
                ax.set_ylim(np.min(medians[:,d]) * 0.97, np.max(medians[:,d]) * 1.03)
            
                ax.plot(np.arange(0,nsteps), medians[:,d], alpha=0.75, lw=0.75, c='darkblue', label='d={:d}'.format(d))
                ax.fill_between(np.arange(0,nsteps), sig_lb[:,d], sig_ub[:,d], color='darkblue', alpha=0.2)
    
    def plot_raw_profile_rr(self, height=6, width=12, num_r=20, num_g=10):
    
        diff_img_r = (self.img_gal_r - self.profile_r) / self.stds_r
        diff_img_r[self.mask] = 0
        
        diff_img_g = (self.img_gal_g - self.profile_g) / self.stds_g
        diff_img_g[self.mask] = 0
    
        fig, ax = plt.subplots(2,4)
        fig.set_figheight(height)
        fig.set_figwidth(width)

        imgs = np.array([self.img_r, self.img_gal_r, self.profile_r, diff_img_r
                         , self.img_g, self.img_gal_g, self.profile_g, diff_img_g])
    
        imgs_min_r = np.min(imgs[:3], axis=(1,2))
        imgs_max_r = np.max(imgs[:3], axis=(1,2))
        
        # lower bound not 0 if negative pixel values in des images not removed
        cbar_lb_r = np.min(imgs_min_r[:2]) - num_r
        cbar_ub_r = np.max(imgs_max_r[:2])
        
        imgs_min_g = np.min(imgs[3:], axis=(1,2))
        imgs_max_g = np.max(imgs[3:], axis=(1,2))
    
        # lower bound not 0 if negative pixel values in des images not removed
        cbar_lb_g = np.min(imgs_min_g[:2]) - num_g
        cbar_ub_g = np.max(imgs_max_g[:2])

        ax[0, 0].set_title(r'DES ID: ' + self.desid
                        , fontsize=16)
        ax[0, 1].set_title(r'$I_{\rm gal}$', fontsize=16)
        ax[0, 2].set_title(r'$I(\mathbf{v})$', fontsize=16)
        ax[0, 3].set_title(r'$\left( I_{\rm gal} - I(\mathbf{v}) \right) / \sigma$', fontsize=16)
    
        # r-band
        
        for i, image in enumerate(imgs[:4]):
            
            # Hide grid lines
            ax[0, i].grid(False)

            # Hide axes ticks
            ax[0, i].set_xticks([])
            ax[0, i].set_yticks([])
    
            if i == 3:
                im = ax[0, i].imshow(image, origin='lower', cmap='RdBu', interpolation='nearest'
                              , vmin=-5, vmax=5
                             )
            
                divider = make_axes_locatable(ax[0, i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label('Residual', rotation=270, labelpad=25, fontsize=16)
        
            else:
                im = ax[0, i].imshow(image, origin='lower', cmap='cubehelix', interpolation='nearest'  # cmap = cubehelix, viridis
                              , vmin=cbar_lb_r, vmax=cbar_ub_r
                             )
            
                #divider = make_axes_locatable(ax[0, i])
                #cax = divider.append_axes('right', size='5%', pad=0.05)
                #cbar = fig.colorbar(im, cax=cax)
                #cbar.set_label(r'$r$-band rightness', rotation=270, labelpad=25, fontsize=16)
         
        # g-band
        
        for i, image in enumerate(imgs[4:]):
            
            # Hide grid lines
            ax[1, i].grid(False)

            # Hide axes ticks
            ax[1, i].set_xticks([])
            ax[1, i].set_yticks([])
    
            if i == 3:
                im = ax[1, i].imshow(image, origin='lower', cmap='RdBu', interpolation='nearest'
                              , vmin=-5, vmax=5
                             )
            
                divider = make_axes_locatable(ax[1, i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label('Residual', rotation=270, labelpad=25, fontsize=16)
        
            else:
                im = ax[1, i].imshow(image, origin='lower', cmap='cubehelix', interpolation='nearest'  # cmap = cubehelix, viridis
                              , vmin=cbar_lb_g, vmax=cbar_ub_g
                             )
            
                #divider = make_axes_locatable(ax[1, i])
                #cax = divider.append_axes('right', size='5%', pad=0.05)
                #cbar = fig.colorbar(im, cax=cax)
                #cbar.set_label(r'$g$-band brightness', rotation=270, labelpad=25, fontsize=16)
        
    def get_medians_calc_amplitude(self):
    
        self.medians = np.array([])

        for i in range(self.dist.shape[2]):
    
            gs1d = self.dist[:,:,i]
    
            gs1d_con = np.concatenate(gs1d)

            mid = np.median(gs1d_con)
    
            self.medians = np.append(self.medians, mid)

        # calculate amplitudes
        self.Ie_r, self.Ie_g = self.norm_to_notnorm(v=self.medians.copy())
    
        return np.append([self.Ie_r, self.Ie_g], self.medians) 
    
    def plot_gal(self, height=6, width=12, num_r=20, num_g=10):
        
        print('----------------------------\n')

        results = self.get_medians_calc_amplitude()
        
        self.Ie_r = results[0]; self.Ie_g = results[1]
        self.medians = results[2:]

        self.x0_med, self.y0_med, self.th_med, self.e_med, self.nr_med, self.rr_med, self.ng_med, self.rg_med = self.medians.copy()
        
        final_params_r = np.array([self.x0_med, self.y0_med, self.th_med, self.e_med, self.nr_med, self.Ie_r, self.rr_med])
        
        self.profile_r = self.imfit_r.getModelImage(newParameters=final_params_r)
        
        final_params_g = np.array([self.x0_med, self.y0_med, self.th_med, self.e_med, self.ng_med, self.Ie_g, self.rg_med])
        
        self.profile_g = self.imfit_g.getModelImage(newParameters=final_params_g)
        
        self.plot_raw_profile_rr(height=height, width=width, num_r=num_r, num_g=num_g)
        
        return self.profile_r, self.profile_g
    
    def calc_mag_auto(self, model):
        '''
        takes a model galaxy image and calculates the galaxy's MAG_AUTO value
        '''
        
        objs = sep.extract(model, 3, err=0.)
        x, y, a, b, theta = objs['x'], objs['y'], objs['a'], objs['b'], objs['theta']
        
        kronrad, krflag = sep.kron_radius(model, x, y, a, b, theta, 6.0)
        rscale = 15 * kronrad

        flux, fluxerr, flag = sep.sum_ellipse(model, x, y, a, b, theta, rscale,
                                              subpix=1)
        mag = -2.5*np.log10(flux) + 30
    
        valid_inds = (np.isfinite(mag) #| np.isnan(mag)
                     )
    
        # if there is more than one mag value and only one is a number, then take mag to be that number
        # if there is more than one mag value and none are numbers, then take mag to be first index
        # if there is more than one mag value and multiple are numbers, then take mag to be the largest number
    
        if (len(mag) > 1) and (len(mag[valid_inds]) == 1):
            mag = mag[valid_inds]
        
        elif (len(mag) > 1) and (len(mag[valid_inds]) > 1):
            mag = mag[valid_inds]
            max_ind = np.argmax(mag)
        
            mag = mag[max_ind]
        
        elif (len(mag) > 1) and (len(mag[valid_inds]) < 1):
            mag = mag[0]
    
        return mag
    
    def mag_arrays(self):
        '''
        calculates MAG_AUTO value for each step in the sampled posterior distribution
        '''
        
        self.rmag_arr = np.full(self.nwalkers * (self.steps - self.burnin), 1.)
        self.gmag_arr = np.full(self.nwalkers * (self.steps - self.burnin), 1.)
        
        try:
            samples = self.dist.reshape(-1, 8)
        
            for ind, sample in tqdm(enumerate(samples)):
            
                x0, y0, th, e, nr, rr, ng, rg = sample
            
                Ie_r, Ie_g = self.norm_to_notnorm(sample)
            
                # r-band
            
                params_r = np.array([x0, y0, th, e, nr, Ie_r, rr])
                profile_r = self.imfit_r.getModelImage(newParameters=params_r)
            
                mag_r = self.calc_mag_auto(profile_r)
                self.rmag_arr[ind] = mag_r
            
                # g-band
            
                params_g = np.array([x0, y0, th, e, ng, Ie_g, rg])
                profile_g = self.imfit_r.getModelImage(newParameters=params_g)
            
                mag_g = self.calc_mag_auto(profile_g)
                self.gmag_arr[ind] = mag_g
            
        except:
            raise NoSamplingError('Posterior distribution must be sampled first.')
    
        return self.rmag_arr, self.gmag_arr
            
# -------------------------------

### Next steps:

#   (DONE) implement magnitude (i.e. MAG_AUTO) measurements using SEP for every posterior PDF evaluation
#             perform measurements on generated model images
#             see this link for info on how to do this: https://sep.readthedocs.io/en/v1.1.x/apertures.html#equivalent-of-flux-auto-e-g-mag-auto-in-source-extractor

#   in the object detection and masking algorithm, add an r-band component to the mask and then test

#   set up a production run in a notebook which does MCMC modeling for a handful of systems
#   find a way to compile the photometric redshift in the process
#   sigma clip outliers in magnitudes just in case (they definitely will show up in g-band at least)
            
# -------------------------------

class NotEnoughStepsError(Exception):
    pass

class NoSamplingError(Exception):
    pass
        