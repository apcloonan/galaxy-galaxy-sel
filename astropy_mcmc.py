### Object Oriented Programming

### Finding Sérsic distribution for DES redMaGiC elliptical galaxies
### using MCMC random sampler

### by Aidan Cloonan

### November 2021

# -------------------------------

### IMPORTS

# math, array manipulation, etc.
import numpy as np

# timing
from timeit import default_timer

## necessary utilities from scipy, astropy and photutils

from scipy.optimize import differential_evolution
from scipy.ndimage import maximum_filter
from scipy.stats import multivariate_normal
from astropy.stats import SigmaClip
from astropy.modeling import functional_models
from astropy.convolution import convolve

# for masking
from photutils.background import Background2D
from photutils.background import StdBackgroundRMS
from photutils.segmentation import deblend_sources
from photutils.segmentation import SegmentationImage

from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.segmentation import detect_sources

# plots
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%matplotlib inline

# MCMC sampling package
import emcee

# -------------------------------

def ellipticity2theta_q(e1, e2):
    """
    transforms complex ellipticity moduli in orientation angle and axis ratio

    :param e1: eccentricity in x-direction
    :param e2: eccentricity in xy-direction
    :return: 1 - axis ratio (minor/major), angle in radian
    """
        
    theta = np.arctan2(e2, e1)/2
        
    c = np.sqrt(e1**2+e2**2)
    c = np.minimum(c, 0.9999)
        
    q = (1-c)/(1+c)
        
    return 1 - q, theta

### MCMC with I_e = 1 optimization

class SersicMCMC:

    def __init__(self, img_r, img_g, std_psf_r, std_psf_g, desid, num=3):
    
        self.img_r = img_r
        self.img_g = img_g
        
        self.desid = desid
        
        self.std_psf_r = std_psf_r
        self.std_psf_g = std_psf_g
        
        self.nrow, self.ncol = self.img_r.shape
        self.xgrid, self.ygrid = np.meshgrid(np.arange(self.ncol), np.arange(self.nrow))
        
        self.params = ['R_{r, 1/2}', 'n_{r}'
                       , 'R_{g, 1/2}', 'n_{g}'
                       , 'x_{0}', 'y_{0}', 'E', '\\theta']
        
        # bounds on posterior PDF sampling
        self.amin, self.amax = 1e-3, 300;
        self.rmin, self.rmax = 0.1, 30;
        self.nmin, self.nmax = 0.1, 8;
        self.x0min, self.x0max = self.ncol/2 - 3, self.ncol/2 + 3;
        self.y0min, self.y0max = self.nrow/2 - 3, self.nrow/2 + 3;
        self.e1min, self.e1max = -0.5, 0.5;
        self.e2min, self.e2max = -0.5, 0.5
        
        self.psf_r, self.psf_g = self.construct_psfs()
        
        img_gal_r, stds_r = self.get_img_uncertainties(self.img_r)
        self.img_gal_r = img_gal_r; self.stds_r = stds_r
        
        img_gal_g, stds_g = self.get_img_uncertainties(self.img_g)
        self.img_gal_g = img_gal_g; self.stds_g = stds_g
        
        masked_img_gal_r, masked_img_gal_g = self.create_mask(num=num)
        
        self.img_gal_r = masked_img_gal_r
        self.img_gal_g = masked_img_gal_g
        
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

    def get_img_uncertainties(self, img_i, num=1.0):
        '''

        a function that estimates uncertainties in the pixel values of redmagic 
        images, both from background and from the elliptical galaxy
    
        also returns the grid of I_gal values, which is what the Sérsic model is fit to
    
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
    
        img_bg = np.mean(bg_pixels)  # I_bg
    
        bkgrms = StdBackgroundRMS(sigma_clip)
        std_bg = bkgrms(img_i) 
    
        ## I_gal
    
        img_gal = img_i - img_bg          # I_tot = I_gal + I_bg
    
        std_gal = np.sqrt(num * img_gal) 
    
        invalid_sgal_ind = (std_gal == np.inf) | (std_gal == -np.inf)
        std_gal[invalid_sgal_ind] = 0
    
        std_gal = np.nan_to_num(std_gal)
    
        ## estimate uncertainties

        std_arr = np.sqrt(std_gal**2 + std_bg**2)

        return img_gal, std_arr
        
    def chi2_sersic(self, v):
    
        ar, rr, nr, ag, rg, ng, x0, y0, e1, e2 = v
        npar = len(v)
        
        e, th = ellipticity2theta_q(e1, e2)
    
        ## initialize Sersic profiles
        
        # r-band
        model_r = functional_models.Sersic2D(amplitude=ar, r_eff=rr, n=nr
                                         , x_0=x0, y_0=y0
                                         , ellip=e, theta=th)
        
        profile_r = model_r(self.xgrid, self.ygrid)
     
        profile_r = convolve(profile_r, self.psf_r)
        
        # g-band
        model_g = functional_models.Sersic2D(amplitude=ag, r_eff=rg, n=ng
                                         , x_0=x0, y_0=y0
                                         , ellip=e, theta=th)
        
        profile_g = model_g(self.xgrid, self.ygrid)
        
        profile_g = convolve(profile_g, self.psf_g)
        
        # calculate E^2, sum of squared errors
        pchi2 = ((self.img_gal_r - profile_r) / self.stds_r)**2 + ((self.img_gal_g - profile_g) / self.stds_g)**2
        
        ### How do I incorporate the g-band into this new formulation?
    
        e2 = np.sum(pchi2)
        norm = pchi2.size - npar
    
        # reduced chi2
        redchi2 = e2 / norm
    
        return redchi2
    
    def chi2_minimize(self, dtypes=None, npop=25):
        '''
        minimizes the chi2 function defined above using a 
        differential evolution algorithm imported from scipy
        
        change to likelihood
        '''
    
        bounds = np.array([(self.amin, self.amax), (self.rmin, self.rmax)                # r-band
                           , (self.nmin, self.nmax)
                           , (self.amin, self.amax), (self.rmin, self.rmax)              # g-band
                           , (self.nmin, self.nmax)
                            , (self.x0min, self.x0max), (self.y0min, self.y0max)
                            , (self.e1min, self.e1max), (self.e2min, self.e2max)])
        
        ar0 = self.amin + (self.amax - self.amin) * np.random.uniform(0,1, size=npop)
        rr0 = self.rmin + (self.rmax - self.rmin) * np.random.uniform(0,1, size=npop)
        nr0 = self.nmin + (self.nmax - self.nmin) * np.random.uniform(0,1, size=npop)
        
        ag0 = self.amin + (self.amax - self.amin) * np.random.uniform(0,1, size=npop)
        rg0 = self.rmin + (self.rmax - self.rmin) * np.random.uniform(0,1, size=npop)
        ng0 = self.nmin + (self.nmax - self.nmin) * np.random.uniform(0,1, size=npop)
        
        x0 = (self.ncol / 2) * np.random.uniform(0.8, 1.2, size=npop)
        y0 = (self.nrow / 2) * np.random.uniform(0.8, 1.2, size=npop)
        e10 = self.e1min + (self.e1max - self.e1max) * np.random.uniform(0,1, size=npop)
        e20 = self.e2min + (self.e2max - self.e2min) * np.random.uniform(0,1, size=npop)
        
        v0 = np.column_stack((ar0, rr0, nr0               # r-band
                              , ag0, rg0, ng0             # g-band
                              , x0, y0, e10, e20       
                             ))
        
        tstart = default_timer()
        
        run = differential_evolution(self.chi2_sersic, popsize=npop, tol = 1e-4
                                     , bounds=bounds
                                     , init = v0
                                    )
    
        print("completed in {:>.5g} sec".format(default_timer() - tstart))
        print("minimum at:",run.x)
        print('f =', run.fun)
    
        return run.x
    
    def create_mask(self, num=3):
        '''
        Takes a raw input fits image and constructs a mask to filter out light 
        from other sources apart from the lens/galaxy. Returns the masked image,
        where pixel values from other sources are set to 0.
    
        '''
    
        # construct a background image w/ background noise
        bkg = Background2D(self.img_g, box_size=5)
        bkg_img = bkg.background
    
        # calculate RMS of each pixel, used to calculate threshold for source identification
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        bkgrms = StdBackgroundRMS(sigma_clip)
        bkgrms_img = bkgrms(self.img_g) 

        # map of thresholds over which sources are detected
        threshold = bkg_img + (0.5 * bkgrms_img)  
    
        # source detection
        sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3).normalize()
        segm = detect_sources(self.img_g, threshold, 5, kernel)
    
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

        segm_dilated_arr = maximum_filter(deblend_copy2.data, num)
        segm_dilated_arr[deblend_copy2.data != 0] = deblend_copy2.data[deblend_copy2.data != 0]
        
        # label central source, which is the lens/galaxy
        segm_dilated_arr[source] = 10000
    
        segm_dilated = SegmentationImage(segm_dilated_arr)
    
        other_inds = np.delete(segm_dilated.labels, -1)
        
        # get pixels from all other sources
        segm_dilated.keep_labels(other_inds)
    
        self.mask = (segm_dilated.data > 0)

        # make values of those pixels 0 in both img_gal and uncertainties
        gal_copy_g = self.img_gal_g.copy()
        gal_copy_g[self.mask] = 0
        gal_copy_r = self.img_gal_r.copy()
        gal_copy_r[self.mask] = 0

        #test_v = self.chi2_minimize()
    
        #ar, rr, nr, ag, rg, ng, x0, y0, e, th = test_v
        
        # use the g-band profile to create the mask
        
        #model = functional_models.Sersic2D(amplitude=ag, r_eff=rg, n=ng
        #                                     , x_0=x0, y_0=y0
        #                                     , ellip=e, theta=th)

        #profile = model(self.xgrid, self.ygrid)

        #cl_residual = (gal_copy_g - profile) / self.stds_g

        # ignore pixels which have already been filtered out
        #cl_residual[nonzeros] = np.nan

        # create a small mask which filters out pixels beyond 3sigma in residual map
        #sigma_clip = SigmaClip(sigma=3, maxiters=10)
        #cl_mask = sigma_clip(cl_residual)

        # turn those into 0s
        #gal_copy_g[cl_mask.mask] = 0
        #gal_copy_r[cl_mask.mask] = 0
    
        return gal_copy_r, gal_copy_g
        
    def log_priors(self, v):
        '''
        all priors are taken to be uniform distributions
        '''
    
        rr, nr, rg, ng, x0, y0, e1, e2 = v
        
        if ((self.rmin <= rr <= self.rmax) and (self.nmin <= nr <= self.nmax)                   # r-band
            and (self.rmin <= rg <= self.rmax) and (self.nmin <= ng <= self.nmax)               # g-band
            and (self.x0min <= x0 <= self.x0max) and (self.y0min <= y0 <= self.y0max) 
            and (self.e1min <= e1 <= self.e1max) and (self.e2min <= e2 <= self.e2max)):
        
            lnprior = 0
            
        else:
        
            lnprior = -np.inf
            
        return lnprior
       
    def log_likelihood(self, v):
    
        self.rr, self.nr, self.rg, self.ng, self.x0, self.y0, self.e1, self.e2 = v
    
        self.e, self.th = ellipticity2theta_q(self.e1, self.e2)
    
        ## initialize Sersic profiles
        
        # r-band
        model_r = functional_models.Sersic2D(amplitude=1.0, r_eff=self.rr, n=self.nr
                                             , x_0=self.x0, y_0=self.y0
                                             , ellip=self.e, theta=self.th)
    
        norm_profile_r = model_r(self.xgrid, self.ygrid)
        norm_profile_r = convolve(norm_profile_r, self.psf_r)
        
        # g-band
        model_g = functional_models.Sersic2D(amplitude=1.0, r_eff=self.rg, n=self.ng
                                             , x_0=self.x0, y_0=self.y0
                                             , ellip=self.e, theta=self.th)
    
        norm_profile_g = model_g(self.xgrid, self.ygrid)
        norm_profile_g = convolve(norm_profile_g, self.psf_r)
    
        ## find amplitudes, then multiply by I'
        
        # r-band
        Ie_r = np.sum((norm_profile_r * (self.img_gal_r)) / (self.stds_r)**2) / np.sum((norm_profile_r)**2 / (self.stds_r)**2)
    
        profile_r = Ie_r * norm_profile_r
        
        # g-band
        Ie_g = np.sum((norm_profile_g * (self.img_gal_g)) / (self.stds_g)**2) / np.sum((norm_profile_g)**2 / (self.stds_g)**2)
    
        profile_g = Ie_g * norm_profile_g
    
        # 1/sqrt(2.*pi) factor can be omitted from the likelihood because it does not depend on model parameters
        # there are two terms for two bands
        return np.sum(-0.5 * (np.log(self.stds_r**2) + (self.img_gal_r - profile_r)**2 / self.stds_r**2)) + np.sum(-0.5 * (np.log(self.stds_g**2) + (self.img_gal_g - profile_g)**2 / self.stds_g**2))
        
    def log_posterior(self, v):
    
        # find values in prior and likelihood pdfs
        logprior = self.log_priors(v)
        loglike = self.log_likelihood(v)
    
        logpost = logprior + loglike
    
        if np.isnan(logpost):
            logpost = -np.inf
    
        return logpost
    
    def mcmc_run(self, ndim=8, nwalkers=25, steps=1250):
        
        min_v = self.chi2_minimize()
        
        noamp_min_v = np.concatenate((min_v[1:3], min_v[4:]))
        
        # initial vector
        # use np.random.normal instead, and center around min_v, small std
        size = ndim * nwalkers
        pr = np.random.normal(loc=0, scale=0.01, size=size).reshape((nwalkers, ndim)) + noamp_min_v
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)

        sampler.run_mcmc(pr, steps, progress=True)
        
        samples = sampler.get_chain()

        for i in range(nwalkers):
            e, th = ellipticity2theta_q(samples[:,i][:,-2], samples[:,i][:,-1])
        
            samples[:,i][:,-2] = e
            samples[:,i][:,-1] = th
            
        self.samples = samples
        
        self.dist = self.samples[-300:]
        
        return self.samples, self.dist
    
    def plot_trace(self, cutoff = 750, figsize=(5, 1)):
        
        try:
            nsteps, ndims = np.shape(self.samples)[0], np.shape(self.samples)[2]
        except:
            nsteps, ndims = np.shape(self.samples)[0], 1
        
        medians = np.median(self.samples, axis=1)
    
        sig_lb = np.percentile(self.samples, 16, axis=1)
        sig_ub = np.percentile(self.samples, 84, axis=1)
        
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
    
        fig, ax = plt.subplots(2,3)
        fig.set_figheight(height)
        fig.set_figwidth(width)

        imgs = np.array([self.img_gal_r, self.profile_r, diff_img_r
                         , self.img_gal_g, self.profile_g, diff_img_g])
    
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

        ax[0, 0].set_title(r'$I_{\rm gal}$, DES ID: ' + self.desid
                        , fontsize=18)
        ax[0, 1].set_title(r'$I(\mathbf{v})$', fontsize=18)
        ax[0, 2].set_title(r'$\left( I_{\rm gal} - I(\mathbf{v}) \right) / \sigma$', fontsize=18)
    
        # r-band
        
        for i, image in enumerate(imgs[:3]):
            
            # Hide grid lines
            ax[0, i].grid(False)

            # Hide axes ticks
            ax[0, i].set_xticks([])
            ax[0, i].set_yticks([])
    
            if i == 2:
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
        
        for i, image in enumerate(imgs[3:]):
            
            # Hide grid lines
            ax[1, i].grid(False)

            # Hide axes ticks
            ax[1, i].set_xticks([])
            ax[1, i].set_yticks([])
    
            if i == 2:
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

        rri, nri, rgi, ngi, x0i, y0i, ei, thi = self.medians.copy()
        
        sersic_r = functional_models.Sersic2D(amplitude=1.0, r_eff=rri, n=nri
                                              , x_0=x0i, y_0=y0i
                                              , ellip=ei, theta=thi)

        sersic_g = functional_models.Sersic2D(amplitude=1.0, r_eff=rgi, n=ngi
                                              , x_0=x0i, y_0=y0i
                                              , ellip=ei, theta=thi)
        
        self.norm_profile_r = sersic_r(self.xgrid, self.ygrid)
    
        # find amplitude
        self.Ie_r = np.sum((self.norm_profile_r * self.img_gal_r) / (self.stds_r)**2) / np.sum((self.norm_profile_r)**2 / (self.stds_r)**2)
        
        self.norm_profile_g = sersic_g(self.xgrid, self.ygrid)
    
        # find amplitude
        self.Ie_g = np.sum((self.norm_profile_g * self.img_gal_g) / (self.stds_g)**2) / np.sum((self.norm_profile_g)**2 / (self.stds_g)**2)
    
        return np.append([self.Ie_r, self.Ie_g], self.medians) 
    
    def plot_gal(self, height=6, width=12, num_r=20, num_g=10):
        
        print('----------------------------\n')

        results = self.get_medians_calc_amplitude()
        
        self.Ie_r = results[0]; self.Ie_g = results[1]
        self.medians = results[2:]

        self.rr_med, self.nr_med, self.rg_med, self.ng_med, self.x0_med, self.y0_med, self.e_med, self.th_med = self.medians.copy()
        
        sersic_r = functional_models.Sersic2D(amplitude=self.Ie_r, r_eff=self.rr_med, n=self.nr_med
                                                , x_0=self.x0_med, y_0=self.y0_med
                                                , ellip=self.e_med, theta=self.th_med)
        
        self.profile_r = sersic_r(self.xgrid, self.ygrid)
        
        sersic_g = functional_models.Sersic2D(amplitude=self.Ie_g, r_eff=self.rg_med, n=self.ng_med
                                                , x_0=self.x0_med, y_0=self.y0_med
                                                , ellip=self.e_med, theta=self.th_med)
        
        self.profile_g = sersic_g(self.xgrid, self.ygrid)
        
        self.plot_raw_profile_rr(height=height, width=width, num_r=num_r, num_g=num_g)

# -------------------------------
# MCMC with analysis on only one filter

class OneBandSersicMCMC:

    def __init__(self, img_r, img_g, desid):
    
        self.img_r = img_r
        self.img_g = img_g
        
        self.desid = desid
        
        self.nrow, self.ncol = self.img_r.shape
        self.xgrid, self.ygrid = np.meshgrid(np.arange(self.ncol), np.arange(self.nrow))
        
        self.params = ['R_{1/2}', 'n', 'x_0', 'y_0', 'E', '\\theta']
        
        # bounds on posterior PDF sampling
        self.amin, self.amax = 1e-3, 300;
        self.rmin, self.rmax = 1e-3, 15;
        self.nmin, self.nmax = 1e-3, 8;
        self.x0min, self.x0max = self.ncol/2 - 3, self.ncol/2 + 3;
        self.y0min, self.y0max = self.nrow/2 - 3, self.nrow/2 + 3;
        self.e1min, self.e1max = -0.5, 0.5;
        self.e2min, self.e2max = -0.5, 0.5
        
        img_gal_r, stds_r = self.get_img_uncertainties(self.img_r)
        self.img_gal_r = img_gal_r; self.stds_r = stds_r
        
        img_gal_g, stds_g = self.get_img_uncertainties(self.img_g)
        self.img_gal_g = img_gal_g; self.stds_g = stds_g
        
        masked_img_gal_r, masked_img_gal_g = self.create_mask()
        
        self.img_gal_r = masked_img_gal_r
        self.img_gal_g = masked_img_gal_g

    def get_img_uncertainties(self, img_i, num=1.0):
        '''

        a function that estimates uncertainties in the pixel values of redmagic 
        images, both from background and from the elliptical galaxy
    
        also returns the grid of I_gal values, which is what the Sérsic model is fit to
    
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
    
        img_bg = np.mean(bg_pixels)  # I_bg
    
        bkgrms = StdBackgroundRMS(sigma_clip)
        std_bg = bkgrms(img_i) 
    
        ## I_gal
    
        img_gal = img_i - img_bg          # I_tot = I_gal + I_bg
    
        std_gal = np.sqrt(num * img_gal) 
    
        invalid_sgal_ind = (std_gal == np.inf) | (std_gal == -np.inf)
        std_gal[invalid_sgal_ind] = 0
    
        std_gal = np.nan_to_num(std_gal)
    
        ## estimate uncertainties

        std_arr = np.sqrt(std_gal**2 + std_bg**2)

        return img_gal, std_arr
        
    def chi2_sersic(self, v):
    
        ar, rr, nr, x0r, y0r, e1r, e2r = v
        npar = len(v)
        
        er, thr = ellipticity2theta_q(e1r, e2r)
    
        # initialize Sersic profile
        model = functional_models.Sersic2D(amplitude=ar, r_eff=rr, n=nr
                                         , x_0=x0r, y_0=y0r
                                         , ellip=er, theta=thr)
        
        profile = model(self.xgrid, self.ygrid)
    
        pchi2 = ((self.img_gal - profile) / self.stds)**2
    
        # E^2, sum of squared errors
        e2 = np.sum(pchi2)
        norm = pchi2.size - npar
    
        # reduced chi2
        redchi2 = e2 / norm
    
        return redchi2
    
    def chi2_minimize(self, dtypes=None, npop=25):
        '''
        minimizes the chi2 function defined above using a 
        differential evolution algorithm imported from scipy
        
        change to likelihood
        '''
    
        bounds = np.array([(self.amin, self.amax), (self.rmin, self.rmax)                # r-band
                           , (self.nmin, self.nmax)
                            , (self.x0min, self.x0max), (self.y0min, self.y0max)
                            , (self.e1min, self.e1max), (self.e2min, self.e2max)])
        
        ar0 = self.amin + (self.amax - self.amin) * np.random.uniform(0,1, size=npop)
        rr0 = self.rmin + (self.rmax - self.rmin) * np.random.uniform(0,1, size=npop)
        nr0 = self.nmin + (self.nmax - self.nmin) * np.random.uniform(0,1, size=npop)
        xr0 = (self.ncol / 2) * np.random.uniform(0.8, 1.2, size=npop)
        yr0 = (self.nrow / 2) * np.random.uniform(0.8, 1.2, size=npop)
        e1r0 = self.e1min + (self.e1max - self.e1max) * np.random.uniform(0,1, size=npop)
        e2r0 = self.e2min + (self.e2max - self.e2min) * np.random.uniform(0,1, size=npop)
        
        v0 = np.column_stack((ar0, rr0, nr0, xr0, yr0, e1r0, e2r0            # r-band
                             ))
        
        tstart = default_timer()
        
        run = differential_evolution(self.chi2_sersic, popsize=npop, tol = 1e-4
                                     , bounds=bounds
                                     , init = v0
                                    )
    
        print("completed in {:>.5g} sec".format(default_timer() - tstart))
        print("minimum at:",run.x)
        print('f =', run.fun)
    
        return run.x
    
    def create_mask(self):
        '''
        Takes a raw input fits image and constructs a mask to filter out light 
        from other sources apart from the lens/galaxy. Returns the masked image,
        where pixel values from other sources are set to 0.
    
        '''
    
        # construct a background image w/ background noise
        bkg = Background2D(self.img_gal_g, box_size=5)
        bkg_img = bkg.background
    
        # calculate RMS of each pixel, used to calculate threshold for source identification
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        bkgrms = StdBackgroundRMS(sigma_clip)
        bkgrms_img = bkgrms(self.img_gal_g) 

        # map of thresholds over which sources are detected
        threshold = bkg_img + (0.5 * bkgrms_img)  
    
        # source detection
        sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3).normalize()
        segm = detect_sources(self.img_gal_g, threshold, 5, kernel)
    
        # deblending sources, looking for saddles between peaks in flux
        segm_deblend = deblend_sources(self.img_gal_g, segm, npixels=5,
                                       nlevels=32, contrast=0.001)
    
        # identify central source, which is the lens/galaxy
        label = segm_deblend.data[(segm.shape[0]//2, segm.shape[1]//2)]
        other_inds = np.delete(np.arange(1, segm_deblend.nlabels+1), label-1)

        # get pixels from all other sources
        segm_deblend_copy = segm_deblend.copy()
        segm_deblend_copy.keep_labels(other_inds)
    
        nonzeros = (segm_deblend_copy.data > 0)

        # make values of those pixels 0 in both img_gal and uncertainties
        gal_copy_g = self.img_gal_g.copy()
        gal_copy_g[nonzeros] = 0
        gal_copy_r = self.img_gal_r.copy()
        gal_copy_r[nonzeros] = 0

        test_v = self.chi2_minimize()
    
        a, r, n, x0, y0, e, th = test_v
        model = functional_models.Sersic2D(amplitude=a, r_eff=r, n=n
                                             , x_0=x0, y_0=y0
                                             , ellip=e, theta=th)

        profile = model(self.xgrid, self.ygrid)

        cl_residual = (gal_copy_g - profile) / self.stds_g

        # ignore pixels which have already been filtered out
        cl_residual[nonzeros] = np.nan

        # create a small mask which filters out pixels beyond 3sigma in residual map
        sigma_clip = SigmaClip(sigma=3, maxiters=10)
        cl_mask = sigma_clip(cl_residual)

        # turn those into 0s
        gal_copy_g[cl_mask.mask] = 0
        gal_copy_r[cl_mask.mask] = 0
    
        return gal_copy_g, gal_copy_r
        
    def log_priors(self, v):
        '''
        all priors are taken to be uniform distributions
        '''
    
        r, n, x0, y0, e1, e2 = v
        
        if ((self.rmin <= r <= self.rmax) and (self.nmin <= n <= self.nmax) 
            and (self.x0min <= x0 <= self.x0max) and (self.y0min <= y0 <= self.y0max) 
            and (self.e1min <= e1 <= self.e1max) and (self.e2min <= e2 <= self.e2max)):
        
            lnprior = 0
            
        else:
        
            lnprior = -np.inf
            
        return lnprior
       
    def log_likelihood(self, v):
    
        self.r, self.n, self.x0, self.y0, self.e1, self.e2 = v
    
        self.e, self.th = ellipticity2theta_q(self.e1, self.e2)
    
        # initialize Sersic profile
        model = functional_models.Sersic2D(amplitude=1.0, r_eff=self.r, n=self.n
                                             , x_0=self.x0, y_0=self.y0
                                             , ellip=self.e, theta=self.th)
    
        norm_profile = model(self.xgrid, self.ygrid)
    
        # find amplitude, then multiply by I'
        Ie = np.sum((norm_profile * (self.img_gal_r + self.img_gal_g)) / (self.stds)**2) / np.sum((norm_profile)**2 / (self.stds)**2)
    
        profile = Ie * norm_profile
    
        # 1/sqrt(2.*pi) factor can be omitted from the likelihood because it does not depend on model parameters
        return np.sum(-0.5 * (np.log(self.stds**2) + (self.img_gal - profile)**2 / self.stds**2))
        
    def log_posterior(self, v):
    
        # find values in prior and likelihood pdfs
        logprior = self.log_priors(v)
        loglike = self.log_likelihood(v)
    
        logpost = logprior + loglike
    
        if np.isnan(logpost):
            logpost = -np.inf
    
        return logpost
    
    def mcmc_run(self, ndim=6, nwalkers=25, steps=800):
        
        min_v = self.chi2_minimize()
        
        # initial vector
        # use np.random.normal instead, and center around min_v, small std
        size = ndim * nwalkers
        pr = np.random.normal(loc=0, scale=0.01, size=size).reshape((nwalkers, ndim)) + min_v[1:]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)

        sampler.run_mcmc(pr, steps, progress=True)
        
        samples = sampler.get_chain()

        for i in range(nwalkers):
            e, th = ellipticity2theta_q(samples[:,i][:,-2], samples[:,i][:,-1])
    
            samples[:,i][:,-2] = e
            samples[:,i][:,-1] = th
            
        self.samples = samples
        
        self.dist = self.samples[-300:]
        
        return self.samples, self.dist
    
    def plot_trace(self, cutoff = 500, figsize=(5, 1)):
        
        try:
            nsteps, ndims = np.shape(self.samples)[0], np.shape(self.samples)[2]
        except:
            nsteps, ndims = np.shape(self.samples)[0], 1
        
        medians = np.median(self.samples, axis=1)
    
        sig_lb = np.percentile(self.samples, 16, axis=1)
        sig_ub = np.percentile(self.samples, 84, axis=1)
        
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
    
    def plot_raw_profile_rr(self):
    
        diff_img = (self.img_gal - self.profile) / self.stds
    
        fig, ax = plt.subplots(1,3)
        fig.set_figheight(4.5)
        fig.set_figwidth(18)

        imgs = np.array([self.img_gal, self.profile, diff_img])
    
        imgs_min = np.min(imgs, axis=(1,2))
        imgs_max = np.max(imgs, axis=(1,2))
    
        # lower bound not 0 if negative pixel values in des images not removed
        cbar_lb = np.min(imgs_min[:2]) - 30
        cbar_ub = np.max(imgs_max[:2])

        ax[0].set_title(r'$I_{\rm gal}$, DES ID: ' + self.desid
                        , fontsize=18)
        ax[1].set_title(r'$I(\mathbf{v})$', fontsize=18)
        ax[2].set_title(r'$\left( I_{\rm gal} - I(\mathbf{v}) \right) / \sigma$', fontsize=18)
    
        for i, image in enumerate(imgs):
            
            # Hide grid lines
            ax[i].grid(False)

            # Hide axes ticks
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    
            if i == 2:
                im = ax[i].imshow(image, origin='lower', cmap='RdBu', interpolation='nearest'
                              , vmin=-5, vmax=5
                             )
            
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label('Residual', rotation=270, labelpad=25, fontsize=16)
        
            else:
                im = ax[i].imshow(image, origin='lower', cmap='cubehelix', interpolation='nearest'  # cmap = cubehelix, viridis
                              , vmin=cbar_lb, vmax=cbar_ub
                             )
            
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label('Brightness', rotation=270, labelpad=25, fontsize=16)
        
    def get_medians_calc_amplitude(self):
    
        self.medians = np.array([])

        for i in range(self.dist.shape[2]):
    
            gs1d = self.dist[:,:,i]
    
            gs1d_con = np.concatenate(gs1d)

            mid = np.median(gs1d_con)
    
            self.medians = np.append(self.medians, mid)

        ri, ni, x0i, y0i, ei, thi = self.medians.copy()
        
        sersic = functional_models.Sersic2D(amplitude=1.0, r_eff=ri, n=ni
                                            , x_0=x0i, y_0=y0i
                                            , ellip=ei, theta=thi)

        self.norm_profile = sersic(self.xgrid, self.ygrid)
    
        # find amplitude
        self.Ie = np.sum((self.norm_profile * self.img_gal) / (self.stds)**2) / np.sum((self.norm_profile)**2 / (self.stds)**2)
    
        return np.append([self.Ie], self.medians) 
    
    def plot_gal(self):
        
        print('----------------------------\n')

        results = self.get_medians_calc_amplitude()
        
        self.Ie = results[0]
        self.medians = results[1:7]

        self.r_med, self.n_med, self.x0_med, self.y0_med, self.e_med, self.th_med = self.medians.copy()
        sersic = functional_models.Sersic2D(amplitude=self.Ie, r_eff=self.r_med, n=self.n_med
                                                , x_0=self.x0_med, y_0=self.y0_med
                                                , ellip=self.e_med, theta=self.th_med)

        self.profile = sersic(self.xgrid, self.ygrid)
        
        self.plot_raw_profile_rr()
        
# -------------------------------
# OLD MCMC VERSION

'''
### MCMC

class SersicMCMC:

    def __init__(self, img, desid):
    
        self.img = img
        
        self.desid = desid
        
        img_gal, stds = self.get_img_uncertainties()
        self.img_gal = img_gal; self.stds = stds
        
        self.nrow, self.ncol = self.img_gal.shape
        self.xgrid, self.ygrid = np.meshgrid(np.arange(self.ncol), np.arange(self.nrow))
        
        self.params = ['A', 'R_{1/2}', 'n', 'x_0', 'y_0', 'E', '\theta']
        
        # bounds on posterior PDF sampling
        self.amin, self.amax = 1e-3, 300;
        self.rmin, self.rmax = 1e-3, 15;
        self.nmin, self.nmax = 1e-3, 8;
        self.x0min, self.x0max = self.ncol/2 - 3, self.ncol/2 + 3;
        self.y0min, self.y0max = self.nrow/2 - 3, self.nrow/2 + 3;
        self.e1min, self.e1max = -0.5, 0.5;
        self.e2min, self.e2max = -0.5, 0.5

    def get_img_uncertainties(self, num=1.0):

        a function that estimates uncertainties in the pixel values of redmagic 
        images, both from background and from the elliptical galaxy
    
        also returns the grid of I_gal values, which is what the Sérsic model is fit to
    
        inputs:
        --------
    
        img        ---    2D array; a DES redMaGiC image with an elliptical galaxy that a Sérsic
                          model is being fit to  
    
        outputs:
        --------
    
        img_gal    ---    2D array; alternatively img_gal_copy (explained below); the grid of I_gal values
                          calculated from estimation of a constant I_bg
    
        std_arr    ---    2D array; grid of uncertainties in corresponding pixel values from the DES image
        
        
    
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    
        bg_pixels = sigma_clip(self.img, masked=False, axis=None)
    
        ## I_bg and std_bg
    
        self.img_bg = np.mean(bg_pixels)  # I_bg
    
        bkgrms = StdBackgroundRMS(sigma_clip)
        std_bg = bkgrms(self.img) 
    
        ## I_gal
    
        img_gal = self.img - self.img_bg          # I_tot = I_gal + I_bg
    
        std_gal = np.sqrt(num * img_gal) 
    
        invalid_sgal_ind = (std_gal == np.inf) | (std_gal == -np.inf)
        std_gal[invalid_sgal_ind] = 0
    
        std_gal = np.nan_to_num(std_gal)
    
        ## estimate uncertainties

        std_arr = np.sqrt(std_gal**2 + std_bg**2)

        return img_gal, std_arr
        
    def chi2_sersic(self, v):
    
        a, r, n, x0, y0, e, th = v
        npar = len(v)
    
        # initialize Sersic profile
        model = functional_models.Sersic2D(amplitude=a, r_eff=r, n=n
                                         , x_0=x0, y_0=y0
                                         , ellip=e, theta=th)
        
        profile = model(self.xgrid, self.ygrid)
    
        pchi2 = ((self.img_gal - profile) / self.stds)**2
    
        # E^2, sum of squared errors
        e2 = np.sum(pchi2)
        norm = pchi2.size - npar
    
        # reduced chi2
        redchi2 = e2 / norm
    
        return redchi2
    
    def chi2_minimize(self, dtypes=None, npop=25):

        minimizes the chi2 function defined above using a 
        differential evolution algorithm imported from scipy


    
        bounds = np.array([(self.amin, self.amax), (self.rmin, self.rmax)
                           , (self.nmin, self.nmax)
                            , (self.x0min, self.x0max), (self.y0min, self.y0max)
                            , (self.e1min, self.e1max), (self.e2min, self.e2max)])
        
        a0 = self.amin + (self.amax - self.amin) * np.random.uniform(0,1, size=npop)
        r0 = self.rmin + (self.rmax - self.rmin) * np.random.uniform(0,1, size=npop)
        n0 = self.nmin + (self.nmax - self.nmin) * np.random.uniform(0,1, size=npop)
        x0 = (self.ncol / 2) * np.random.uniform(0.8, 1.2, size=npop)
        y0 = (self.nrow / 2) * np.random.uniform(0.8, 1.2, size=npop)
        e10 = self.e1min + (self.e1max - self.e1max) * np.random.uniform(0,1, size=npop)
        e20 = self.e2min + (self.e2max - self.e2min) * np.random.uniform(0,1, size=npop)
        
        v0 = np.column_stack((a0, r0, n0, x0, y0, e10, e20))
        
        tstart = default_timer()
        
        run = differential_evolution(self.chi2_sersic, popsize=npop, tol = 1e-4
                                     , bounds=bounds
                                     , init = v0
                                    )
    
        print("completed in {:>.5g} sec".format(default_timer() - tstart))
        print("minimum at:",run.x)
        print('f =', run.fun)
    
        return run.x
        
    def log_priors(self, v):

        all priors are taken to be uniform distributions

    
        a, r, n, x0, y0, e1, e2 = v
        
        if ((self.amin <= a <= self.amax) and (self.rmin <= r <= self.rmax) 
            and (self.nmin <= n <= self.nmax) 
            and (self.x0min <= x0 <= self.x0max) and (self.y0min <= y0 <= self.y0max) 
            and (self.e1min <= e1 <= self.e1max) and (self.e2min <= e2 <= self.e2max)):
        
            lnprior = 0
            
        else:
        
            lnprior = -np.inf
            
        return lnprior
       
    def log_likelihood(self, v):
    
        self.a, self.r, self.n, self.x0, self.y0, self.e1, self.e2 = v
    
        self.e, self.th = ellipticity2theta_q(self.e1, self.e2)
    
        # initialize Sersic profile
        model = functional_models.Sersic2D(amplitude=self.a, r_eff=self.r, n=self.n
                                             , x_0=self.x0, y_0=self.y0
                                             , ellip=self.e, theta=self.th)
        
        self.profile = model(self.xgrid, self.ygrid)
    
        # 1/sqrt(2.*pi) factor can be omitted from the likelihood because it does not depend on model parameters
        return np.sum(-0.5 * (np.log(self.stds**2) + (self.img_gal - self.profile)**2 / self.stds**2))
        
    def log_posterior(self, v):
    
        # find values in prior and likelihood pdfs
        logprior = self.log_priors(v)
        loglike = self.log_likelihood(v)
    
        logpost = logprior + loglike
    
        if np.isnan(logpost):
            logpost = -np.inf
    
        return logpost
    
    def mcmc_run(self, ndim=7, nwalkers=25):
        
        min_v = self.chi2_minimize()
        
        # initial vector
        pr = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim)) + (min_v - 0.5)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior)

        sampler.run_mcmc(pr, 800, progress=True)
        
        samples = sampler.get_chain()

        for i in range(nwalkers):
            e, th = ellipticity2theta_q(samples[:,i][:,-2], samples[:,i][:,-1])
    
            samples[:,i][:,-2] = e
            samples[:,i][:,-1] = th
            
        self.samples = samples
        
        self.dist = self.samples[-300:]
        
        return self.samples, self.dist
    
    def plot_trace(self, cutoff = 500, figsize=(10, 3)):
        
        try:
            nsteps, ndims = np.shape(self.samples)[0], np.shape(self.samples)[2]
        except:
            nsteps, ndims = np.shape(self.samples)[0], 1
        
        medians = np.median(self.samples, axis=1)
    
        sig_lb = np.percentile(self.samples, 16, axis=1)
        sig_ub = np.percentile(self.samples, 84, axis=1)
        
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
    
    def plot_raw_profile_rr(self):
    
        diff_img = (self.img_gal - self.profile) / self.stds
    
        fig, ax = plt.subplots(1,3)
        fig.set_figheight(4.5)
        fig.set_figwidth(18)

        imgs = np.array([self.img_gal, self.profile, diff_img])
    
        imgs_min = np.min(imgs, axis=(1,2))
        imgs_max = np.max(imgs, axis=(1,2))
    
        # lower bound not 0 if negative pixel values in des images not removed
        cbar_lb = np.min(imgs_min[:2]) - 40
        cbar_ub = np.max(imgs_max[:2])

        ax[0].set_title(r'$I_{\rm gal}$, DES ID: ' + self.desid
                        , fontsize=18)
        ax[1].set_title(r'$I(\mathbf{v})$', fontsize=18)
        ax[2].set_title(r'$\left( I_{\rm gal} - I(\mathbf{v}) \right) / \sigma$', fontsize=18)

        ax[0].set_ylabel(r'$y$', fontsize=18)
    
        for i, image in enumerate(imgs):
    
            if i == 2:
                im = ax[i].imshow(image, origin='lower', cmap='RdBu', interpolation='nearest'
                              , vmin=-5, vmax=5
                             )
            
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label('Residual', rotation=270, labelpad=25, fontsize=16)
        
            else:
                im = ax[i].imshow(image, origin='lower', cmap='cubehelix', interpolation='nearest'  # cmap = cubehelix, viridis
                              , vmin=cbar_lb, vmax=cbar_ub
                             )
            
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label('Brightness', rotation=270, labelpad=25, fontsize=16)
            
            ax[i].tick_params(axis='both', which='major', labelsize=16)
            ax[i].tick_params(axis='both', which='minor', labelsize=16)
            ax[i].set_xlabel(r'$x$', fontsize=18)
        
    def plot_gal(self):
        
        print('----------------------------\n')

        medians = np.array([])

        for i in range(self.dist.shape[2]):
    
            gs1d = self.dist[:,:,i]
    
            gs1d_con = np.concatenate(gs1d)
    
            mid = np.median(gs1d_con)
    
            medians = np.append(medians, mid)

        self.a_med, self.r_med, self.n_med, self.x0_med, self.y0_med, self.e_med, self.th_med = medians.copy()
        sersic = functional_models.Sersic2D(amplitude=self.a_med, r_eff=self.r_med, n=self.n_med
                                                , x_0=self.x0_med, y_0=self.y0_med
                                                , ellip=self.e_med, theta=self.th_med)

        self.profile = sersic(self.xgrid, self.ygrid)
        
        self.plot_raw_profile_rr()
'''