# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:38:52 2026

@author: asp
"""

import os
import numpy as np
from astropy.io import fits

lam_min = 950e-9
lam_stp = 50e-9
lam_itv = 18 
lam_lst = np.arange(lam_min,lam_min+(lam_itv)*lam_stp+1e-9,lam_stp)
nL = lam_lst.shape[0]

# 500Hz
rcn_data = 'd:Andes/Data_corono/data/OPDs_PASSATA/OPD/WS/500HzVarWS/'

dirs = ('20250704_105833.0', '20250704_112007.0',
        '20250704_114101.0', '20250704_121550.0',
        '20250704_123540.0', '20250704_125507.0',
        '20250704_131433.0', '20250704_135325.0',
        '20250704_141253.0', '20250707_114321.0')

# 1kHz
# rcn_data = 'd:Andes/Data_corono/data/OPDs_PASSATA/OPD/WS/1kHzVarWS/'

# dirs = [('1', '2',
#         '3', '4',
#         '5', '6',
#         '7', '8',
#         '9', '10')]

ndirs = len(dirs)


donow = '20260715161753' # 500Hz
# donow = '20251001085445' # 1kHz

rcn_simu = 'd:Andes/Data_corono/'
out = rcn_simu +  'results/' + donow + '/'


pup = fits.getdata('d:Andes/Data_corono/data/Pupil/ELT_pupil_400.fits')
iok = np.nonzero(pup)

s_avg = np.zeros((nL, ndirs))
m_avg = np.zeros((nL, ndirs))

# for d in range(10):
for d, opds in enumerate(dirs):
    
    # opd = fits.getdata(rcn_data + 'OPDs_PASSATA/OPD/WS/1kHzVarWS/'+str(d+1)+
    #                    '/CUBE_CL_coo0.0_0.0-00'+str(d+1)+'.fits')
    opd_set = os.path.basename(rcn_data + opds).split('.')[0]
    
    flist_opd = []
    for fnm in os.listdir(rcn_data + opds):
        if fnm.endswith(".fits"):
            flist_opd.append(fnm)
    
    flist_opd = sorted(flist_opd,key=len)
    
    opd = fits.getdata(rcn_data + dirs[d] + '/' + flist_opd[0])
    # opd = fits.getdata(rcn_data + 'OPDs_PASSATA/OPD/WS/500HzVarWS/'+'20250704_105833.0'+'/CUBE_CL_coo0.0_0.0.fits')
    
    start = int((opd.shape)[2]/5)
    opd = opd[:,:,start:]
    
    opd_shape = opd.shape
    strehl = np.zeros((nL, opd_shape[2]))
    marechal = np.zeros((nL , opd_shape[2]))
    
    opd *= 1e-9
    opd *= pup[:,:,None]
    
    for l in range(nL):

        s_tmp = np.mean(np.exp(2j*np.pi*opd[iok]/lam_lst[l]),axis=0)
        strehl[l,:] = np.abs(s_tmp)**2.

        marechal[l,:] = np.exp(-np.std(
            opd[iok]*2.*np.pi/lam_lst[l], axis=0)**2.)
            
        s_avg[l,d] = np.mean(strehl[l,:])    
        m_avg[l,d] = np.mean(marechal[l,:])   
        
        print(str(d+1)+': \t', np.round(lam_lst[l]*1e6,3), '\t', 
              np.round(s_avg[l,d], 3), '\t', np.round(m_avg[l,d], 3))
    
    hdr = fits.getheader(out + opd_set + '/ao_corr_psf_' + donow + '.fits')
    fits.writeto( out + opd_set + '/strehl.fits',
                 np.asarray([lam_lst, s_avg[:,d]]), hdr, overwrite=True )
    fits.writeto( out + opd_set + '/marechal.fits',
                 np.asarray([lam_lst, m_avg[:,d]]), hdr, overwrite=True )

    
print('moyenne globale: ' , np.transpose([
    np.round(lam_lst*1e6,3),
    np.round(np.mean(s_avg, axis=1),3),
    np.round(np.mean(m_avg, axis=1),3)]))

# manual hdr ...
hdr = fits.getheader(out + opd_set + '/ao_corr_psf_' + donow + '.fits')
fits.writeto( out + '/strehl_all_sets.fits',
             np.asarray([lam_lst, np.mean(s_avg, axis=1)]), 
             hdr, overwrite=True )
fits.writeto( out + '/marechal_all_sets.fits',
             np.asarray([lam_lst, np.mean(m_avg, axis=1)]), 
             hdr, overwrite=True )

