# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:12:09 2024

@author: asp
"""


import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
from astropy.io import fits
import os
from pathlib import Path
# from mpl_toolkits.axes_grid1 import AxesGrid
from datetime import datetime  #  asp for datetime of now

# from psf_profile import profile

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!

donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  asp, datetime of now

# conversion l radian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas


#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits data
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots

elif user == 'Adrien':
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/results/').resolve()
    fdir_plt   = Path(
        '/Users/asimonnin/Desktop/PhD/Andes/Data_corono/plots/').resolve()

elif user == 'Mamadou':
    fdir_base = ("/Users/mndiaye/Library/CloudStorage/"\
                 "OneDrive-UniversitéNiceSophiaAntipolis/data/andes")
    # Directory for the OPD with the corresponding seed value
    fdir_res   = Path( fdir_base / 'results' ).resolve()
    fdir_plt   = Path( fdir_base / 'plots' ).resolve()

#yjh
# jeu de données avec erreur de 2.pi dans la dispersion: 80 est 80/2.pi, etc.
# was_donow = ('20241007141137','20241007141603','20241009084549',
#              '20241009084821','20241011173119')

# 0 dispersion directory first
#yjh 0, 5, 10, 15 mas/µ disp 
# was_donow = ('20241007141137','20241106094210',
#               '20241024101416','20241106094322')

# yjh 0, 10, 20, 30, 40, 50 nm rms ncpa
# was_donow = ('20241007141137','20241105085056',
#               '20241105131506','20241105155837',
#               '20241105155850')
#,'20241105155859')

# offset in lambda central/D for psf to fpm + yjh
# 0 to 0.5, step 0.05
# was_donow = ('20241007141137','20241112095915',
#              '20241112100035','20241112100137',
#              '20241112100549','20241112100648')

# 0 to 1, step 0.25
was_donow = ('20241007141137','20241112100648',
              '20241112175847','20241112175935','20241112180014')

as_oi = 25.

was_donow = was_donow[::-1]

#%%

for was_d in was_donow:
    
    fdir_res2 = fdir_res / was_d

    cfile = fdir_res2/('contrast_JQM_'+was_d+'.fits')
    data = fits.getdata(cfile)        # contrast mean
    # data2 = fits.getdata(cfile,ext=1)   # contrast std     
    data3 = fits.getdata(cfile,ext=2)   # gain

    sdir = os.listdir(fdir_res2)[0]
    hfile = os.listdir((fdir_res2/sdir))[0]
    head = fits.getheader((fdir_res2/sdir/hfile))
    try:
        disp = np.round(head['DISP']*1e-6,1)
    except KeyError:
        disp=-1
    
    try:
        ncpa = head['NCPA']
    except KeyError:
        ncpa=-1
    try:
        offset = head['FDEC']
    except KeyError:
        offset=0
    
    lamC = head['LMBD']
    
    lam_min = head['LMIN']
    lam_stp = head['LSTP']
    lam_itv = head['LITV']
    lam_lst = np.arange(lam_min,lam_min+(lam_itv+1)*lam_stp,lam_stp)
    nL = len(lam_lst)
        
    nImg = head['NIMG']
    D = head['DIAM']
    mB = head['SFPM']
    
    offset *= D/lamC
    offset = np.round(offset,2)
    
    aS = np.arange(nImg//2)*(mas2rad * 58.393 / (D *1e9) ) # ang. sep.
    pos_as_oi = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))

    plt.figure(1, (8, 4.5))
    plt.tight_layout()
    plt.xlabel(r'$\lambda$ (nm)') 
    #[$\lambda$/D]')
    plt.ylabel(f'Contrast @ {int(as_oi)} mas')
    plt.yscale('log')

    plt.title(r'Impact of tilt, median condition (0"65)')
    # plt.title(r'Impact of ncpa, median condition (0"65)')
    # plt.title(r'Impact of dispersion, median condition (0"65)')

    plt.grid(True)

    if offset==0 and (disp==0 or disp==-1) and (ncpa==0 or ncpa==-1):
                
        # plt.plot(lam_lst*1e9, data3[0,:]*data[0,:],color='black' , label='atmo., no coro.', ls='--')
        atmoNoCoro = data3[0,:]*data[0,:]                
            
    # plt.plot(lam_lst*1e9, data[0,:], label=str(disp))
    plt.plot(lam_lst*1e9, data[0,:], label=f'{offset*lamC*mas2rad/D:.1f} mas')
    # plt.plot(lam_lst*1e9, data[0,:], label=f'{ncpa} nm rms')
    # plt.plot(lam_lst*1e9, data[0,:], label=f'{disp} mas/µm')

    # plt.plot(lam_lst*1e9, data[0,:], label=f'{offset*lamC*mas2rad/D:.1f} mas')
    # plt.plot(lam_lst*1e9, data[0,:], label=f'{offset*lamC*mas2rad/D:.1f} mas')
    # plt.fill_between(lam_lst*1e9, data[0,:] - data2[0,:], data[0,:] + data2[0,:], alpha=0.2)


bfiles = os.listdir((fdir_res2/'perfect'))
bfile = bfiles[np.argmax((lambda x:[len(i) for i in x])(bfiles))]
bdata = fits.getdata(fdir_res2/'perfect'/bfile)
   
plt.plot(lam_lst*1e9, bdata[:,pos_as_oi], color='black', ls=':')  #  , label='no atmo., coro.')
# plt.plot(lam_lst*1e9, atmoNoCoro,color='black' , label='atmo., no coro.', ls='--')

plt.ylim(3e-5,3e-1)
plt.legend(fontsize='small', loc=2, ncol=1)

plt.plot(lam_lst*1e9, atmoNoCoro,color='black' , ls='--')

#♠, label='atmo., no coro.')
bbox = dict(boxstyle='square', fc='w', alpha=0.125)
plt.text(2000,1e-1,'--- atmo., no coro.', bbox=bbox)
plt.text(2000,1e-4,'... no atmo., coro.', bbox=bbox)


# fname_contrast_25mas = ('all_windshake_data_contrast_25mas_asRatioOf_lbd2D_ringAvgdPrfs_vs_wvl_' +
#                     os.path.basename(file_cro[0]).split('.')[0])
#if actual lambda/D
# fname_contrast_25mas = ('all_windshake_data_contrast_' + ptrn +'_'+as_str+'mas_'+ was_donow)
# fpath_contrast_25mas_svg = fdir_plt / (fname_contrast_25mas + '.svg')
# fpath_contrast_25mas_pdf = fdir_plt / (fname_contrast_25mas + '.pdf')
# plt.savefig(fpath_contrast_25mas_svg)
# plt.savefig(fpath_contrast_25mas_pdf)

plt.savefig('C:/Users/asp/Desktop/tilt.svg')
plt.savefig('C:/Users/asp/Desktop/tilt.pdf')
plt.savefig('C:/Users/asp/Desktop/tilt.png')


plt.show()
    
    