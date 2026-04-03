# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:02:59 2026

@author: asp
"""

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from astropy.io import fits
import pyzelda.utils.zernike as zernike

# from PYZELDA


def disc(dim, size, diameter=False, strict=False, center=(), cpix=False, invert=False, mask=False):
    '''
    Create a numerical array containing a disc.

    Parameters
    ----------
    dim : int
        Size of the output array

    size : int
        Size of the disk, representing either the radius (default) or the diameter

    diameter : bool, optional
        Specify if input size is the diameter. Default is 'False'

    strict : bool optional
        If set to Trye, size must be strictly less than (<), instead of less
        or equal (<=). Default is 'False'

    center : sequence, optional
        Specify the center of the disc. Default is '()', i.e. the center of the array

    cpix : bool optional
        If set to True, the disc is centered on pixel at position (dim//2, dim//2).
        Default is 'False', i.e. the disc is centered between 4 pixels

    invert : bool, optinal
        Specify if the disc must be inverted. Default is 'False'

    mask : bool, optional
        Specify if return value is a masked array or a numerical array. Default
        is 'False'

    Returns
    -------
    disc : array
        An array containing a disc with the specified parameters
    '''

    if (diameter is True):
        rad = size/2
    else:
        rad = size

    if (len(center) == 0):
        if (cpix is False):
            cx = (dim-1) / 2
            cy = (dim-1) / 2
        else:
            cx = dim // 2
            cy = dim // 2
    elif (len(center) == 2):
        cx = center[0]
        cy = center[1]
    else:
        raise ValueError('Error, you must pass 2 values for center')
        return None

    x = np.arange(dim, dtype=np.float64) - cx
    y = np.arange(dim, dtype=np.float64) - cy
    xx, yy = np.meshgrid(x, y)

    rr = np.sqrt(xx**2 + yy**2)

    if (strict is True):
        msk = (rr < rad)
    else:
        msk = (rr <= rad)

    if (invert is True):
        msk = np.logical_not(msk)

    if (mask is True):
        return msk
    else:
        d = np.zeros((dim, dim))
        d[msk] = 1

    return d



def compute_fft_opd(opd, mask=None, freq_cutoff=None):
    '''
    Compute the fft of the opd normalized in physical units (nm/cycle_per_pupil)

    Parameters
    ----------
    opd : array_like
        OPD map in nanometers

    mask : array_like
        Pupil mask

    freq_cutoff : float
        Maxium spatial frequency of the psd

    Returns
    -------
    fft_opd: array_like
        Normalized fft of the opd
    '''

    Dpup = opd.shape[-1]
    dim = 2 ** (np.ceil(np.log(2 * Dpup) / np.log(2)))
    # sampling = dim / Dpup

    # compute the surface of the mask pupil
    if mask is None:
        norm = np.sqrt(1 / ((Dpup ** 2) * np.pi / 4))
    else:
        opd = opd * mask
        norm = np.sqrt(1 / mask.sum())

    # compute psd with fft or mft
    if freq_cutoff is None:
        pad_width = int((dim - Dpup) / 2)
        pad_opd = np.pad(opd, pad_width, 'constant')
        fft_opd = norm * fft.fftshift(fft.fft2(fft.fftshift(pad_opd), norm='ortho'))
    # else:
    #     fft_opd = norm * mft.mft(opd, Dpup, int(2*freq_cutoff*sampling), 2*freq_cutoff)

    return fft_opd


def compute_psd(opd, mask=None, freq_cutoff=None):
    '''
    Compute the power spectral density fro a given phase map

    When freq_Cutoff is specified, psd is computed with mft, using the
    same sampling as the fft that would normally be used to compute
    the psd.  This smapling makes the computation of the normalization
    factor consistent with the standard fft case.
    Parameters
    ----------
    opd : array_like
        OPD map in nanometers

    mask : array_like
        Pupil mask

    freq_cutoff : float
        Maxium spatial frequency of the psd

    Returns
    -------
    psd_2d: array_like
        PSD map

    psd_1d: vector
        Azimuthal averaged profile of the PSD map

    freq: vector
        Vector of spatial frequencies corresponding to psd_1d
    '''

    # Dpup = opd.shape[-1]
    # dim = 2 ** (np.ceil(np.log(2 * Dpup) / np.log(2)))
    # sampling = dim / Dpup

    # remove piston
    if mask is not None:
        idx = (mask != 0)
        opd[idx] -= opd[idx].mean()

    fft_opd = compute_fft_opd(opd, mask, freq_cutoff)
    psd_2d = np.abs(fft_opd) ** 2
    # psd_1d, rad = prof.mean(psd_2d)

    # compute psd with fft or mft
    # if freq_cutoff is None:
    #     freq = rad * Dpup / dim
    # else:
    #     freq = rad / sampling

    return psd_2d #○ , psd_1d, freq


def integrate_psd(psd_2d, freq_cutoff, freq_min, freq_max):
    '''
    Compute the integration of the psd between two spatial frequency bounds

    Parameters
    ----------
    psd_2d: array_like
        PSD map normalized in (nm/cycle per pupil)^2

    freq_cutoff : float
        Maxium spatial frequency of the psd

    freq_min : float
        Lower bound of the spatial frequencies for integration

    freq_max : float
        Upper bound of the spatial frequencies for integration

    Returns
    -------
    sigma : float
        Integrated value of the psd in nanometers

    '''

    dim = psd_2d.shape[-1]
    freq_min_pix = freq_min * dim / (2 * freq_cutoff)
    freq_max_pix = freq_max * dim / (2 * freq_cutoff)

    if freq_min == 0:
        disk = disc(dim, freq_max_pix, diameter=False)
    else:
        disk = disc(dim, freq_max_pix, diameter=False) \
               - disc(dim, freq_min_pix, diameter=False)

    sigma = np.sqrt(psd_2d[disk == 1].sum())
    
    return sigma


if __name__ == "__main__":
    
    fnm = r'D:\Andes\Data_corono\data\ncpa_ELT_pupil_400_30nm_4096screens.fits'
    
    ncpa = fits.getdata(fnm)[0:3999,:,:] * 1e9
    # ipup = np.nonzero(ncpa[10,:,:])
    mask = ((ncpa[0,:,:] - ncpa[0,0,0]) != 0 ) * 1.

    ncpa_shp = ncpa.shape
    nPup = ncpa_shp[-1]
    
        # number of Zernike modes
    nZern = 10
    nDim = nPup + 2
    
    # crop the pupil to have array size equal to the pupil diameter
    ini = (nDim-nPup)//2
    end = (nDim+nPup)//2
    
    """
    ### Generation of the Zernike modes
    """
    
    Zern_arr0 = np.zeros((nZern, nDim, nDim))
    for i in range(nZern):
        Zern_arr0[i] = zernike.zernike1(i+2,npix=nDim)
        Zern_arr0[i, np.isnan(Zern_arr0[i])] = 0.
        
    Zern_arr = np.zeros((nZern, nPup, nPup))
    # crop the pupil to have array size equal to the pupil diameter
    Zern_arr = Zern_arr0[:, ini:end, ini:end]
    plt.imshow(Zern_arr[-1])

    res = np.zeros((nZern,ncpa_shp[0]))
    
    for j in range(ncpa_shp[0]):
        
        for i in range(nZern):
        
            res[i,:] = np.sum(ncpa[j,:,:]*mask*Zern_arr[i]) / np.sum(Zern_arr[i]*Zern_arr[i])
        
    print(np.mean(res,axis=1))
    print(np.sqrt(np.sum(np.mean(res,axis=1)**2.)))
    print(np.sqrt(30**2. - np.sum(np.mean(res,axis=1)**2.))) # 30: nm RMS ncpa

    
    # kx = (np.arange(nPup)-nPup//2) + 0.5
    # ky = (np.arange(nPup)-nPup//2) + 0.5
    # kx2, ky2 = np.meshgrid(kx, ky)
    # rho = np.sqrt(kx2**2 + ky2**2)
    
    # zn1 = np.nonzero(np.where(rho<=2.,1.,0.))
    # zn2 = np.nonzero(np.where(np.abs(rho-(7.5+2.)/2.)<(7.5-2.)/2.,1.,0.))
    # zn3 = np.nonzero(np.where(rho>7.5,1.,0))
        
    # for i in range(ncpa.shape[0]):
        
    #     # psd = (np.fft.fftshift(np.fft.fft2(
    #     #     np.fft.ifftshift(ncpa[i,:,:]),norm="ortho")))
        
    #     psd = compute_fft_opd(ncpa[i,:,:], mask)
        
    #     psd = np.abs(psd)**2.
        
    #     res[:,i] = [integrate_psd(psd, 199.5, 0, 2),
    #                 integrate_psd(psd, 199.5, 2, 7.5),
    #                 integrate_psd(psd, 199.5, 7.5, 199.5),
    #                 integrate_psd(psd, 199.5, 0, 199.5)]

    # print('\n', np.mean(res[0,:]), '\n', np.mean(res[1,:]),
    #       '\n', np.mean(res[2,:]), '\n', np.mean(res[3,:]))

