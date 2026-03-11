# -*- coding: utf-8 -*-
"""
@author: Blake Ledger, updated June 3, 2025

This is a compilation of functions from Nikki Zabel's KILOGAS
GitHub page and what was used for VERTICO. I kept what I felt
was needed for the KILOGAS smooth and clip, and changed the
code and functions to remove steps which were not necessary.

Some additional updates and changes, including implementing
the Dame+ 2011 smooth+clip method, by Tim Davis on April 17, 2025.
"""

from astropy.io import fits
import numpy as np
from scipy.ndimage import binary_dilation, label,uniform_filter
import os

class KILOGAS_clip:

    def __init__(self, gal, path_pbcorr, path_uncorr, start, stop, verbose, save, read_path, save_path, sun_method_params=[3,3,2,2, None, 0.1, None, None, None],dame_method_params=[5,1.5,4,1], spec_res=10, pb_thresh=40, prune_by_npix=None):
        self.galaxy = gal
        self.path_pbcorr = path_pbcorr
        self.path_uncorr = path_uncorr
        self.start = start
        self.stop = stop
        self.nchan_low = sun_method_params[0]
        self.cliplevel_low = sun_method_params[1]
        self.nchan_high = sun_method_params[2]
        self.cliplevel_high = sun_method_params[3]
        self.prune_by_npix = prune_by_npix
        self.prune_by_fracbeam = sun_method_params[5]
        self.expand_by_npix = sun_method_params[6]
        self.expand_by_fracbeam = sun_method_params[7]
        self.expand_by_nchan = sun_method_params[8]
        self.dame_clipsn=dame_method_params[0]
        self.dame_beamexpand=dame_method_params[1]
        self.dame_chanexpand=dame_method_params[2]
        self.dame_suppress_subbeam_artifacts=dame_method_params[3]
        self.verbose = verbose
        self.tosave = save
        self.readpath = read_path
        self.savepath = save_path
        self.spec_res = spec_res
        self.pb_thresh = pb_thresh

        
    def do_clip(self, method='dame'):
        """
        Perform the clipping of the data cube, using either the Dame+11
        standard smooth and clip or the more complicated explanding mask
        method adopted and optimised by Jiayi Sun.

        Parameters
        ----------
        method (Default 'dame' method)
            The smooth and clip method to implement (code is optimized for
            the Dame+11 method, implemented by Tim Davis and updated by here
            by Blake Ledger)

        Returns
        -------
        FITS file
            The clipped and trimmed version of the input data cube.
        FITS file
            The corresponding noise cube, of the same dimensions as the output
            data cube.

        """
        
        if self.verbose:
            print("DO CLIP")

        ## read in the fits files
        cube_pbcorr, cube_uncorr = self.readfits()    
        cube_uncorr_copy = cube_uncorr.copy()

        if self.verbose:
            print("     Number of channels in initial cube:", len(cube_pbcorr.data))

        # Spit the cube in an emission and noise part
        emiscube_pbcorr = cube_pbcorr.data[self.start:self.stop, :, :]
        emiscube_uncorr = cube_uncorr.data[self.start:self.stop, :, :]

        noisecube_pbcorr = np.concatenate((cube_pbcorr.data[:self.start, :, :],
                                           cube_pbcorr.data[self.stop:, :, :]), axis=0)
        noisecube_uncorr = np.concatenate((cube_uncorr_copy.data[:self.start, :, :],
                                           cube_uncorr_copy.data[self.stop:, :, :]), axis=0)

        if self.verbose:
            print("     The first channel of the CO line:", self.start)
            print("     The last channel of the CO line:", self.stop)
            print("     Number of channels in the split cube:", len(emiscube_pbcorr))
        
        emiscube_uncorr_hdu = fits.PrimaryHDU(emiscube_uncorr, cube_uncorr_copy.header)
        noisecube_uncorr_hdu = fits.PrimaryHDU(noisecube_uncorr, cube_uncorr_copy.header)
        emiscube_pbcorr_hdu = fits.PrimaryHDU(emiscube_pbcorr, cube_pbcorr.header)
        noisecube_pbcorr_hdu = fits.PrimaryHDU(noisecube_pbcorr, cube_pbcorr.header)
        
        if method == 'sun':
            mask = self.sun_method(emiscube_uncorr_hdu, noisecube_uncorr_hdu, len(cube_pbcorr.data))
        elif method == 'dame':
            mask = self.dame_method(emiscube_uncorr_hdu, noisecube_uncorr_hdu, len(cube_pbcorr.data),
                                   emiscube_pbcorr_hdu)

        # Mask spaxels under a certain threshold of pb response
        mask = self.mask_pb(mask, emiscube_pbcorr, emiscube_uncorr)

        # Prune small island to get rid of remaining pb noise
        #mask = self.prune_small_detections(emiscube_uncorr_hdu, mask)
            
        mask_hdu = fits.PrimaryHDU(mask.astype(int), cube_pbcorr.header)

        if self.verbose:
            print("MASK MADE")
        
        if self.tosave:
            if self.verbose:
                print('     Saving mask')
                
            if method == 'sun':    
                mask_hdu.header.add_comment('Cube was clipped using the Sun+18 masking method', before='BUNIT')
            elif method == 'dame':
                mask_hdu.header.add_comment('Cube was clipped using the Dame+11 masking method', before='BUNIT')
            
            try:
                mask_hdu.header.pop('BTYPE')
                mask_hdu.header.pop('BUNIT')
            except:
                pass
            try:
                mask_hdu.header.pop('DATAMAX')
                mask_hdu.header.pop('DATAMIN')
                mask_hdu.header.pop('JTOK')
                mask_hdu.header.pop('RESTFRQ')
            except:
                pass

            if method == 'sun':
                mask_hdu.header['CLIP_RMS'] = self.sun_method(emiscube_uncorr_hdu, noisecube_uncorr_hdu, len(cube_pbcorr.data), calc_rms=True)
            elif method == 'dame':
                mask_hdu.header['CLIP_RMS'] = self.dame_method(emiscube_uncorr_hdu, noisecube_uncorr_hdu, len(cube_pbcorr.data), emiscube_pbcorr_hdu, calc_rms=True)
            mask_hdu.header.comments['CLIP_RMS'] = 'rms [K km/s] for clipping'

            # Adjust the header to match the velocity range used
            mask_hdu.header['CRVAL3'] += self.start * mask_hdu.header['CDELT3']

            if self.spec_res == 10:
                if not os.path.exists(self.savepath + 'by_galaxy/' + self.galaxy):
                    os.makedirs(self.savepath + 'by_galaxy/' + self.galaxy, exist_ok=True)
                if not os.path.exists(self.savepath + 'by_galaxy/' + self.galaxy + '/10kms'):
                    os.makedirs(self.savepath + 'by_galaxy/' + self.galaxy + '/10kms', exist_ok=True)
                mask_hdu.writeto(self.savepath + 'by_galaxy/' + self.galaxy + '/10kms/' + self.galaxy + '_mask_cube.fits', overwrite=True) 

                if not os.path.exists(self.savepath + 'by_product/cubes'):
                    os.makedirs(self.savepath + 'by_product/cubes', exist_ok=True)
                if not os.path.exists(self.savepath + 'by_product/cubes/' + '10kms'):
                    os.makedirs(self.savepath + 'by_product/cubes/' + '10kms', exist_ok=True)
                mask_hdu.writeto(self.savepath + 'by_product/cubes/' + '10kms/' + self.galaxy + '_mask_cube.fits', overwrite=True)
            elif self.spec_res == 30:
                if not os.path.exists(self.savepath + 'by_galaxy/' + self.galaxy):
                    os.makedirs(self.savepath + 'by_galaxy/' + self.galaxy, exist_ok=True)
                if not os.path.exists(self.savepath + 'by_galaxy/' + self.galaxy + '/30kms'):
                    os.makedirs(self.savepath + 'by_galaxy/' + self.galaxy + '/30kms', exist_ok=True)
                mask_hdu.writeto(self.savepath + 'by_galaxy/' + self.galaxy + '/30kms/' + self.galaxy + '_mask_cube.fits', overwrite=True) 

                if not os.path.exists(self.savepath + 'by_product/cubes'):
                    os.makedirs(self.savepath + 'by_product/cubes', exist_ok=True)
                if not os.path.exists(self.savepath + 'by_product/cubes/' + '30kms'):
                    os.makedirs(self.savepath + 'by_product/cubes/' + '30kms', exist_ok=True)
                mask_hdu.writeto(self.savepath + 'by_product/cubes/' + '30kms/' + self.galaxy + '_mask_cube.fits', overwrite=True)

        emiscube_pbcorr[mask == 0] = 0
        clipped_hdu = fits.PrimaryHDU(emiscube_pbcorr, cube_pbcorr.header)

        # Adjust the header to match the velocity range used
        clipped_hdu.header['CRVAL3'] += self.start * clipped_hdu.header['CDELT3']
        
        if self.verbose:
            print("CLIP APPLIED")
        self.add_clipping_keywords(emiscube_uncorr_hdu, noisecube_uncorr_hdu, len(cube_pbcorr.data), clipped_hdu.header)

        if self.tosave:
            if self.verbose:
                print("EMISSION CUBE SAVED")
            if self.spec_res == 10:
                clipped_hdu.writeto(self.savepath+ 'by_galaxy/' + self.galaxy + '/10kms/' + self.galaxy + '_clipped_cube.fits', overwrite=True)
                clipped_hdu.writeto(self.savepath+ 'by_product/cubes/10kms/' + self.galaxy + '_clipped_cube.fits', overwrite=True)
            elif self.spec_res == 30:
                clipped_hdu.writeto(self.savepath+ 'by_galaxy/' + self.galaxy + '/30kms/' + self.galaxy + '_clipped_cube.fits', overwrite=True)
                clipped_hdu.writeto(self.savepath+ 'by_product/cubes/30kms/' + self.galaxy + '_clipped_cube.fits', overwrite=True)
        
        return clipped_hdu, noisecube_pbcorr_hdu
        
    def readfits(self):
        """
        Read in the fits files containing the primary beam corrected and uncorrected specral cubes.
        
        Parameters
        ----------
        None

        Returns
        -------
        cube_pbcorr : FITS file
            FITS file containing the spectral cube with primary beam correction applied and the header.
        cube_uncorr : FITS file
            FITS file containing the spectral cube without primary beam correction applied and the header.
        """
        
        if self.verbose:
            print("READ .FITS")

        cube_pbcorr = fits.open(self.path_pbcorr)[0]
        cube_uncorr = fits.open(self.path_uncorr)[0]            

        # If the first and last channels consist of nans only, remove them
        spectrum_pbcorr = np.nansum(cube_pbcorr.data, axis=(1, 2))
        cube_pbcorr.data = cube_pbcorr.data[spectrum_pbcorr != 0, :, :]
        
        spectrum_uncorr = np.nansum(cube_uncorr.data, axis=(1, 2))
        cube_uncorr.data = cube_uncorr.data[spectrum_uncorr != 0, :, :]

        # Count the number of empty channels at the start of the cube, so the header can be corrected
        firstzero = np.nonzero(spectrum_pbcorr)[0][0]
        cube_pbcorr.header['CRVAL3'] += cube_pbcorr.header['CDELT3'] * firstzero
        cube_uncorr.header['CRVAL3'] += cube_uncorr.header['CDELT3'] * firstzero

        # Get rid of nans
        cube_pbcorr.data[~np.isfinite(cube_pbcorr.data)] = 0
        cube_uncorr.data[~np.isfinite(cube_uncorr.data)] = 0

        return cube_pbcorr, cube_uncorr

    def dame_method(self, emiscube, noisecube, size, emiscube_pbcorr=None, calc_rms=False):
        """
        Function added by Tim Davis on April 17, 2025
        Function updated by Blake Ledger and Scott Wilkinson on June 3, 2025
        
        Apply a uniform smooth, using sigma = 3 (updated from = 4 by Blake) in the velocity direction
        (seems to work best), to the uncorrected cube.
        :return: (ndarray) mask to apply to the un-clipped cube
        """
        
        if self.verbose and not(calc_rms):
            print('DO DAME METHOD')
            
        if calc_rms:
            if self.verbose:
                print('     Add rms value to header')
            return np.nanstd(self.innersquare(noisecube.data))
        
        cellsize = emiscube.header['CDELT2']  # deg. / pix.
        bmaj_pix = emiscube.header['BMAJ']/cellsize  # deg. 
        bmin_pix = emiscube.header['BMAJ']/cellsize  # deg.  
            
        sigma = self.dame_beamexpand * bmaj_pix
        
        smooth_cube = uniform_filter(emiscube.data, size=[self.dame_chanexpand, sigma, sigma], mode='constant') 
        smooth_cube_noise = uniform_filter(noisecube.data, size=[self.dame_chanexpand, sigma, sigma], mode='constant') 

        #newrms = np.nanstd(self.innersquare(smooth_cube_noise)[self.start-10:self.start+10,:,:])
        #New noise calculation fixed to included cases where line is within 10 channels of the start or end of the cube
        
        if (self.start<=10) and ((size - self.stop)>20):
            # if line starts within 10 channels from the left edge, use the 20 channels immediately following the line
            noise_window = self.innersquare(smooth_cube_noise)[self.start:self.start+20,:,:]
        
        elif ((size - self.stop)<=10) and (self.start>20):
            # if line end is 10 channels from the right edge, use the 20 channels immediately preceding the line
            noise_window = self.innersquare(smooth_cube_noise)[self.start-20:self.start,:,:]
        
        elif ((size - self.stop)>10) and (self.start>10):
            # if line is 10 channels or more from either edge, take the 10 channels on either side
            noise_window = self.innersquare(smooth_cube_noise)[self.start-10:self.start+10,:,:]

        else:
            print('WARNING: The line starts and ends within ten channels of the edges of the cube. Consider expanding your frequency range.')
            noise_window = None
        
        if noise_window is not None:
            newrms = np.nanstd(noise_window)
        else:
            newrms = np.nan

        if emiscube_pbcorr is not None:
            pb_cube = emiscube.data / emiscube_pbcorr.data
        else:
            pb_cube = np.ones_like(emiscube)
        self.maskcliplevel = newrms * self.dame_clipsn / np.abs(pb_cube)
        mask=(smooth_cube > self.maskcliplevel)
        
        if self.dame_suppress_subbeam_artifacts:
            labels,cnt=label(mask)
            hist,lab=np.histogram(labels,bins=np.arange(cnt+1))        
            beam_area_pix = np.pi * bmaj_pix * bmin_pix
            for thelabel in lab[0:-1][hist<(beam_area_pix*self.dame_suppress_subbeam_artifacts)]:
                mask[labels == thelabel]=False
        
        return mask

    def innersquare(self, cube):
        """
        Get the central square (in spatial directions) of the spectral cube (useful for calculating the
        rms in a PB corrected spectral cube). Can be used for 2 and 3 dimensions, in the latter case the
        velocity axis is left unchanged.
        
        Parameters
        ----------
        cube : 2D or 3D array
            3D array input cube or 2D image
            
        Returns
        -------
        cube : 2D or 3D array
            2D or 3D array of the inner 1/8 of the cube in the spatial directions
        """

        if self.verbose:
            print('MEASURE RMS NOISE IN INNERSQUARE')
        
        if len(cube.shape) == 3:
            start_x = int(cube.shape[1] / 2 - cube.shape[1] / 8)
            stop_x = int(cube.shape[1] / 2 + cube.shape[1] / 8)
            start_y = int(cube.shape[2] / 2 - cube.shape[1] / 8)
            stop_y = int(cube.shape[2] / 2 + cube.shape[1] / 8)
            inner_square = cube[:, start_x:stop_x, start_y:stop_y]
            
            if (inner_square == inner_square).any():
                return inner_square
            else:
                return cube

        elif len(cube.shape) == 2:
            start_x = int(cube.shape[0] / 2 - 20)
            stop_x = int(cube.shape[0] / 2 + 20)
            start_y = int(cube.shape[1] / 2 - 20)
            stop_y = int(cube.shape[1] / 2 + 20)
            inner_square = cube[start_x:stop_x, start_y:stop_y]
            if (inner_square == inner_square).any():
                return inner_square
            else:
                return cube
            
        else:
            raise AttributeError('Please provide a 2D or 3D array.')
    
    def add_clipping_keywords(self, emiscube, noisecube, size, header, method='dame'):
        """
        Add information to the header specifying details about the clipping.

        Parameters
        ----------
        header : FITS header
            Header of the cube that was clipped.

        Returns
        -------
        header : FITS header
            Header of the cube with clipping-related keywords added.

        """

        if self.verbose:
            print('     Add keywords to cube header')

        if method == 'dame':
            try:
                header.add_comment('Cube was clipped using the Dame+11 masking method', before='BUNIT')
            except:
                header.add_comment('Cube was clipped using the Dame+11 masking method', after='NAXIS2')
            header['D_CLIP'] = self.dame_clipsn
            header.comments['D_CLIP'] = 'S/N threshold in mask'
            header['D_B_EXP'] = self.dame_beamexpand
            header.comments['D_B_EXP'] = 'Beam expand factor in mask'
            header['D_C_EXP'] = self.dame_chanexpand
            header.comments['D_C_EXP'] = 'Channel expand factor in mask'
            header['PRUNE'] = self.dame_suppress_subbeam_artifacts
            header.comments['PRUNE'] = 'Prune by fraction factor in mask'

            header['CLIP_RMS'] = self.dame_method(emiscube, noisecube, size, calc_rms=True)
            header.comments['CLIP_RMS'] = 'rms [K km/s] for clipping'

        elif method == 'sun':
            try:
                header.add_comment('Cube was clipped using the Sun+18 masking method', before='BUNIT')
            except:
                header.add_comment('Cube was clipped using the Sun+18 masking method', after='NAXIS2')
            header['CLIPL_L'] = self.cliplevel_low
            header.comments['CLIPL_L'] = 'S/N threshold specified for "wing mask"'
            header['CLIPL_H'] = self.cliplevel_high
            header.comments['CLIPL_H'] = 'S/N threshold specified for "core mask"'
            header['NCHAN_L'] = self.nchan_low
            header.comments['NCHAN_L'] = '# of consecutive channels for "core mask"'
            header['NCHAN_H'] = self.nchan_high
            header.comments['NCHAN_H'] = '# of consecutive channels for "wing mask"'
    
            header['CLIP_RMS'] = self.sun_method(emiscube, noisecube, size, calc_rms=True)
            header.comments['CLIP_RMS'] = 'rms [K km/s] for clipping'

        return header
            
    def sun_method(self, emiscube, noisecube, size, calc_rms=False):
        """
        Apply Jiayi Sun's clipping method, including options to prune 
        detections with small areas on the sky, or expand the mask in the mask
        along the spatial axes or spectral axis.

        Parameters
        ----------
        emiscube : FITS file
            3D cube containing the spectral line.
        noisecube : FITS file
            3D cube containing the line-free channels.
        calc_rms : bool, optional
            If set to "True" this function will only return the RMS estimate 
            for the data cube. The default is False.

        Raises
        ------
        AttributeError
            Will raise an AttributeError if some of the required parameters
            are not set (nchan_low, cliplevel_low, nchan_high, and 
                         cliplevel_high).

        Returns
        -------
        mask : FITS file
            The final mask, which will be used for clipping the original cube.

        """

        if self.verbose and not(calc_rms):
            print('DO SUN METHOD')
            
        # Check if the necessary parameters are provided
        if not (
                self.nchan_low and self.cliplevel_low and self.nchan_high and
                self.cliplevel_high):
            
            raise AttributeError('If you want to use Sun\'s method, please provide "nchan_low", "cliplevel_low", '
                                 '"nchan_high", and "cliplevel_high" and the rest of the sun_method_params.')

        # Estimate the rms from the spatial inner part of the cube
        inner = self.innersquare(noisecube.data)
        #rms = np.nanstd(inner[self.start-10:self.start+10,:,:])
        rms = np.nanstd(inner)
        
        if calc_rms:
            if self.verbose:
                print('     Add rms value to header')
                
            return rms

        snr = emiscube.data / rms

        # Generate core mask
        mask_core = (snr > self.cliplevel_high).astype(bool)
        for i in range(self.nchan_high - 1):
            mask_core &= np.roll(mask_core, shift=1, axis=0)
        mask_core[:self.nchan_high - 1] = False
        for i in range(self.nchan_high - 1):
            mask_core |= np.roll(mask_core, shift=-1, axis=0)

        # Generate wing mask
        mask_wing = (snr > self.cliplevel_low).astype(bool)
        for i in range(self.nchan_low - 1):
            mask_wing &= np.roll(mask_wing, shift=1, axis=0)
        mask_wing[:self.nchan_low - 1] = False
        for i in range(self.nchan_low - 1):
            mask_wing |= np.roll(mask_wing, shift=-1, axis=0)

        # Dilate core mask inside wing mask
        mask = binary_dilation(mask_core, iterations=0, mask=mask_wing)

        # Prune detections with small projected areas on the sky
        if self.prune_by_fracbeam or self.prune_by_npix:
            mask = self.prune_small_detections(emiscube, mask)

        # Expand along spatial dimensions by a fraction of the beam FWHM
        if self.expand_by_fracbeam or self.expand_by_npix:
            mask = self.expand_along_spatial(emiscube, mask)

        # Expand along spectral dimension by a number of channels
        if self.expand_by_nchan:
            mask = self.expand_along_spectral(mask)

        return mask

    def prune_small_detections(self, cube, mask):
        """
        Mask structures in the spectral cube that are smaller than the desired 
        size specified by "prune_by_npix" or "prune_by_fracbeam" in the galaxy 
        parameters. Based on the function designed by Jiayi Sun.

        Parameters
        ----------
        cube : FITS file
            The ALMA cube, used to extract the relevant beam information from 
            the header.
        mask : 3D numpy array
            The mask that we previously created from the smoothed data cube.

        Returns
        -------
        mask : 3D numpy array
            Updated mask with the small detections set to 0.

        """

        if self.verbose:
            print("CREATE PRUNING MASK")
            
        if self.prune_by_npix:
            prune_by_npix = self.prune_by_npix
        else:
            res = cube.header['CDELT2']  # deg. / pix.
            bmaj_pix = cube.header['BMAJ'] / res  # deg. / (deg. / pix.)
            bmin_pix = cube.header['BMIN'] / res  # deg. / (deg. / pix.)
            beam_area_pix = np.pi * bmaj_pix * bmin_pix
            prune_by_npix = beam_area_pix * self.prune_by_fracbeam

        labels, count = label(mask)
        for idx in np.arange(count) + 1:
            if (labels == idx).any(axis=0).sum() < prune_by_npix:
                mask[labels == idx] = False

        return mask
    
    def expand_along_spatial(self, cube, mask):
        """
        Expand the mask along spatial dimensions by an amount provided by 
        either "expand_by_npix" or "expand_by_fracbeam" in the galaxy 
        parameters.

        Parameters
        ----------
        cube : FITS file
            The ALMA cube, used to extract the relevant beam information from 
            the header.
        mask : 3D numpy array
            The mask that we previously created from the smoothed data cube.

        Returns
        -------
        mask : 3D numpy array
            Updated, expanded mask with the additional pixels set to 1.

        """

        if self.verbose:
            print("SPATIAL MASK EXPANSION")

        if self.expand_by_npix:
            expand_by_npix = int(self.expand_by_npix)
        else:
            res = cube.header['CDELT2']  # deg. / pix.
            bmaj = cube.header['BMAJ']  # deg.
            bmin = cube.header['BMIN']  # deg.
            beam_hwhm_pix = np.average([bmaj, bmin]) / res / 2  # deg. / (deg. / pix.)
            expand_by_npix = int(beam_hwhm_pix * self.expand_by_fracbeam)

        structure = np.zeros([3, expand_by_npix * 2 + 1, expand_by_npix * 2 + 1])
        Y, X = np.ogrid[:expand_by_npix * 2 + 1, :expand_by_npix * 2 + 1]
        R = np.sqrt((X - expand_by_npix) ** 2 + (Y - expand_by_npix) ** 2)
        structure[1, :] = R <= expand_by_npix
        mask = binary_dilation(mask, iterations=1, structure=structure)

        return mask
    
    def expand_along_spectral(self, mask):
        """
        Expand the mask along the velocity direction as provided by 
        "expand_by_nchan" in the galaxy parameters.

        Parameters
        mask : 3D numpy array
            The mask that we previously created from the smoothed data cube.

        Returns
        -------
        mask : 3D numpy array
            Updated, expanded mask with the additional pixels set to 1.

        """

        if self.verbose:
            print("SPECTRAL MASK EXPANSION")

        for i in range(self.expand_by_nchan):
            tempmask = np.roll(mask, shift=1, axis=0)
            tempmask[0, :] = False
            mask |= tempmask
            tempmask = np.roll(mask, shift=-1, axis=0)
            tempmask[-1, :] = False
            mask |= tempmask

        return mask


    def mask_pb(self, mask, cube_pbcorr, cube_uncorr):

        # Mask spaxels below a certain threshold of the pb response
        pb_cube = cube_pbcorr.copy()
        pb_cube = cube_uncorr / cube_pbcorr
        mask[pb_cube < self.pb_thresh / 100] = 0
        
        return mask
        
        




    


