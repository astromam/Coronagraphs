# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:36:32 2024

@author: asp
"""

    psf_ratio = Int_D0_psf_avg / Int_D_psf_avg
    fname_psf_ratio = ('pixel2pixel_gain_map_'
                       + os.path.basename(file_cro[-1]).split('.')[0]+'.fits')
    fpath_psf_ratio =  fdir_res / opd_set / fname_psf_ratio 
    fits.writeto(fpath_psf_ratio, psf_ratio, head_psf, overwrite=True)
  
            ratio_prf[i,p] = np.mean(psf_ratio[i,:,:][ring_val])
