# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:04:18 2025

@author: asp
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

as_oi = 15.

# wavelengths in m
lamC = 1600e-9  #  some reference wvl unique value
lam_min = 960e-9  #  min value in range
lam_max = 2450e-9  #  max value in range  #  2450e-9 // 1800e-9
lam_itv = 18  #  nb of intervals in range --> nb+1 wvl's !  #  18 // 10
lam_stp = np.floor(np.ceil((lam_max-lam_min)*1e9/lam_itv)/10)*1e-8  # wvl step
lam_lst = np.arange(lam_min,lam_max,lam_stp) if lam_max != lam_min else [lamC]
nL = len(lam_lst)

# Pupil diameter in m 
D = 38.54

# conversion l radian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

lamCD2mas = (lamC/D)*mas2rad

#head_psf = fits.getheader(res_dir / file_psf[0])

nImg = 400
pscale = 0.3
hlf_fov = nImg * pscale / 2.
   
aS = np.arange(nImg//2)*(mas2rad * hlf_fov / (D *1e9) ) # ang. sep.
pos_as_oi = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))

wd = 'd:Andes/Data_corono/results/'
# files=('20250317172539/contrast_JQ1_20250317172539.fits',
#        '20250318154752/contrast_JQ1_20250318154752.fits',
#        '20250318154814/contrast_JQ1_20250318154814.fits',
#        '20250318154832/contrast_JQ1_20250318154832.fits',
#        '20250319083152/contrast_JQ1_20250319083152.fits',
#        '20250319083224/contrast_JQ1_20250319083224.fits')

files=('20250317172539/contrast_JQ1_20250317172539_15.fits',
       '20250319083224/contrast_JQ1_20250319083224_15.fits')

plt.figure(0, (8, 4.5))
plt.tight_layout()
plt.xlabel(r'Wavelength $\lambda$ [nm]') #[$\lambda$/D]')
plt.ylabel(f'Contrast @ {int(as_oi)} mas')
plt.yscale('log')
plt.ylim(1e-5,1e-1)
plt.title('Ideal coronagraph')
plt.grid(True)

# scl_fac = (100,90,80,70,60,50)
rmsValue=('100','50')
for i, fnm in enumerate(files):
    
    data = fits.getdata(wd+fnm)
    dat2 = fits.getdata(wd+fnm,ext=1)
    plt.plot(lam_lst*1e9, data[0,:], label=rmsValue[i]+' nm RMS')
    plt.fill_between(lam_lst*1e9, dat2[0,:], dat2[1,:], alpha=0.2)
    
plt.legend(title='AO residuals',fontsize='small', loc=4)
plt.savefig(
    wd+'../plots/idealCoronagraphVsAOresiduals100and50nmRMS_15mas-20032025.pdf',
    bbox_inches='tight', pad_inches=0.1)  
plt.show()    

                                                       