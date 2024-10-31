# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:12:09 2024

@author: asp
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
import os
from pathlib import Path
from mpl_toolkits.axes_grid1 import AxesGrid
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
# was_donow = ('20241007141137','20241024101416')
# was_donow = ('20241007141137','20241025140035','20241028101351',
#              '20241028161658','20241029090244','20241029131750',
#              '20241029160850')
was_donow = ('20241007141137','20241030082457',
             '20241030161024','20241031105129')

#hk
# jeu de données avec erreur de 2.pi dans la dispersion: 80 est 80/2.pi, etc.
# was_donow = ('20241009164421','20241009165029','20241010080453',
#              '20241010140433','20241010140448')

as_oi = 25.

#%%

once = True

for was_d in was_donow:
    
    fdir_res2 = fdir_res / was_d

    cfile = fdir_res2/('contrast_JQM_'+was_d+'.fits')
    data = fits.getdata(cfile)        # contrast mean
    data2 = fits.getdata(cfile,ext=1)   # contrast std     
    data3 = fits.getdata(cfile,ext=2)   # gain

    sdir = os.listdir(fdir_res2)[0]
    hfile = os.listdir((fdir_res2/sdir))[0]
    head = fits.getheader((fdir_res2/sdir/hfile))
    disp = np.round(head['DISP']*1e-6,1)
    ncpa = head['NCPA']
    
    lam_min = head['LMIN']
    lam_stp = head['LSTP']
    lam_itv = head['LITV']
    lam_lst = np.arange(lam_min,lam_min+(lam_itv+1)*lam_stp,lam_stp)
    nL = len(lam_lst)
        
    nImg = head['NIMG']
    D = head['DIAM']
    mB = head['SFPM']
    
    aS = np.arange(nImg//2)*(mas2rad * 58.393 / (D *1e9) ) # ang. sep.
    pos_as_oi = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))

    plt.figure(1, (8, 6))
    plt.tight_layout()
    plt.xlabel(r'$\lambda$ (nm)')#[$\lambda$/D]')
    plt.ylabel('Contrast')
    plt.yscale('log')
    plt.title('<contrast>@25 mas & ncpa (nm rms),\n median res. atm. (0"65)')
    plt.grid(True)

    if once:
        
        bfiles = os.listdir((fdir_res2/'perfect'))
        bfile = bfiles[np.argmax((lambda x:[len(i) for i in x])(bfiles))]
        bdata = fits.getdata(fdir_res2/'perfect'/bfile)   
        
        plt.plot(lam_lst*1e9, bdata[:,pos_as_oi], color='black', label='coro, no atm.')
        plt.plot(lam_lst*1e9, data3[0,:]*data[0,:],color='black' , label='atm. only, no coro', ls='--')

        once = False
    
    # plt.plot(lam_lst*1e9, data[0,:], label=str(disp))
    plt.plot(lam_lst*1e9, data[0,:], label=str(ncpa))
    # plt.fill_between(lam_lst*1e9, data[0,:] - data2[0,:], data[0,:] + data2[0,:], alpha=0.2)

plt.ylim(2e-5,2e-1)
plt.legend(fontsize='small', loc=2, ncol=2)



# fname_contrast_25mas = ('all_windshake_data_contrast_25mas_asRatioOf_lbd2D_ringAvgdPrfs_vs_wvl_' +
#                     os.path.basename(file_cro[0]).split('.')[0])
#if actual lambda/D
# fname_contrast_25mas = ('all_windshake_data_contrast_' + ptrn +'_'+as_str+'mas_'+ was_donow)
# fpath_contrast_25mas_svg = fdir_plt / (fname_contrast_25mas + '.svg')
# fpath_contrast_25mas_pdf = fdir_plt / (fname_contrast_25mas + '.pdf')
# plt.savefig(fpath_contrast_25mas_svg)
# plt.savefig(fpath_contrast_25mas_pdf)


plt.show()
    
    