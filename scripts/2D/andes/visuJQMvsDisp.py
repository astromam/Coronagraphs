# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:12:09 2024

@author: asp
"""


import numpy as np
import matplotlib.pyplot as plt
plt.close()

from astropy.io import fits
import os
from pathlib import Path
from datetime import datetime  #  asp for datetime of now

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
# directory with no error first in the tuple, then monotically increasing

#  0, 10, 20, 30, 40, 50 nm rms ncpa nm rms with yjh 0mas/µ disp
# was_donow = ('20250121142815','20250122183859','20250122184044',
#              '20250122184053','20250122184101','20250122184109')

# 0.89/0.35/3.9 + ncpa 
# was_donow = ('20251020152206','20251016135837','20251016135818',
#              '20251016135714','20251001085445')

# 20251016135714 10 nm RMS
# 20251016135818 30 nm RMS
# 20251016135837 50 nm RMS
# 20251020152206 70 nm RMS

# 0 dispersion directory first
#yjh 0, 5, 10, 15 mas/µm disp 
# was_donow = ('20250121142815','20250124135415',
#              '20250124135527','20250124135557')

# 0.89/0.35/3.9 + disp
# was_donow = ("20251020152303","20251015162408","20251014150115",
#              "20251014150042","20251001085445")

# 20251014150042  5 mas/µ
# 20251014150115 10 mas/µ
# 20251015162408 15 mas/µ
# 20251020152303 20 mas/µ

# lyot stop angular position error
# 0.9/0.37/4.0
# was_donow = ('20250121142815','20250214100748',
#              '20250214100828','20250214100857')

# 0.89 0.35 3.9 + LS angular position error
# was_donow = ('20251017120423','20251017115610','20251017115549',
#              '20251017115345','20251001085445')
# 20251017115345 0.5°
# 20251017115549 1.0°
# 20251017115610 1.5°
# 20251017120335 2.0°
# 20251017120423 3.0°
# 20251017120448 4.0°

# offset/tilt of 0, 1, 2, 3 and 4 mas for psf to fpm + yjh
# was_donow = ('20250121142815','20250217154848','20250217155004',
#               '20250217155048','20250217155138')
 
# was_donow = ("20251015134855","20251013163616","20251013163502",
#               "20251013163430","20251001085445")

# lyot stop vertical (elevation) offset in pixels: 1, 2, 4, 8
# 2024->05/2025 09/037/4.0
# was_donow = ('20250121142815',"20250130104616","20250130104747",
#               "20250130104759","20250130104809")

# 0.89/0.35/3.9 + Lyot stop vertical offset
# 20251104110334 2 pixels / 0.5% D
# 20251104110517 4 pixels / 1.0% D
# 20251104110556 8 pixels / 2.0% D
# was_donow = ("20251104110556","20251104110517",
#               "20251104110334","20251001085445")

# lyot stop horizontal (azimuth) offset in pixels: 1, 2, 4, 8
# was_donow = ('20250121142815','20250214100537','20250214100614',
#              '20250214100649','20250214100716')


# 0.89/0.35/3.9 + Lyot stop azimutal offset
# 20251107134453 2 pixels / 0.5% D 
# 20251107134538 4 pixels / 1.0% D 
# 20251107134615 8 pixels / 2.0% D 

was_donow = ("20251107134615","20251107134538",
              "20251107134453","20251001085445")
# fpm defocus in nm RMS: 10, 20, 30, 40, 50
# was_donow = ('20250121142815',"20250210175500","20250210175517",
#              "20250210175531","20250210175546","20250210175606")

# 0.89/0.35/3.9 + fpm defocus
# 20251103103334  10 nm RMS
# 20251103103358  30 nm RMS
# 20251103103429  50 nm RMS
# 20251105104900  70 nm RMS

# was_donow = ('20251105104900','20251103103429','20251103103358',
#              '20251103103334','20251001085445')

jq = 'JQM'
seeings = {'JQ1':'0.43"', 'JQ2':'0.58"','JQM':'0.65"','JQ3':'0.74"','JQ4':'1.06"'}
jqs=seeings.get(jq)

as_oi = 20.
spxl = 7.
# was_donow = was_donow[::-1]

#%%
c_tab= plt.cm.inferno(np.linspace(.85,0.,len(was_donow))) # ['c','b','k','r','m','g','y']
#c_tab[-1]=[0.,0.,0.,0.]


for i, was_d in enumerate(was_donow):
    
    fdir_res2 = fdir_res / was_d

    cfile = fdir_res2/('contrast_'+str(int(as_oi))+'mas_'+
                       str(int(spxl))+'mas_'+jq+'_'+was_d+'.fits')
    print('cfile:', cfile)
    data = fits.getdata(cfile)        # contrast mean
    data2 = fits.getdata(cfile,ext=1)   # contrast min/max     
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
 
    try:
        ls_ape = head['LSAE']
    except KeyError:
        ls_ape=0
 
    try:
        fpm_dfe_elt = head['ELT_DFOC']
    except KeyError:
        fpm_dfe_elt=0.


    lsvoe = 0
    try:
        ls_voe = head['LSVE']
    except KeyError:
        ls_voe=0
        try:
            ls_voe = head['LSOE']
        except KeyError:
            ls_voe=0

    try:
        ls_hoe = head['LSHE']
    except KeyError:
        ls_hoe=0


    lamC = head['LMBD']
    
    lam_min = head['LMIN']
    lam_stp = head['LSTP']
    lam_itv = head['LITV']
    lam_lst = np.arange(lam_min,lam_min+(lam_itv+1)*lam_stp,lam_stp)
    lam_lst = lam_lst[np.where(lam_lst < 1860e-9)]

    nL = len(lam_lst)
        
    nPup = head['NPUP']
    nImg = head['NIMG']
    D = head['DIAM']
    mB = head['SFPM']
    diam = head['FDIA']
    
    offset *= D/lamC
    offset = np.round(offset,2)
    
    aS = np.arange(nImg//2)*(mas2rad * 58.393 / (D *1e9) ) # ang. sep.
    pos_as_oi = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))

    plt.figure(1, (8, 4.5))
    plt.tight_layout()
    plt.xlabel(r'Wavelength $\lambda$ [nm]') 
    #[$\lambda$/D]')
    plt.ylabel(f'Contrast @ {int(as_oi)} mas')
    plt.yscale('log')

    plt.title(f'Coronagraph config. in YJH band, {jqs} seeing ({jq})')
    
    plt.grid(True)

    if (offset==0 and (disp==0 or disp==-1) and (ncpa==0 or ncpa==-1)
        and ls_ape==0 and ls_voe==0 and ls_hoe==0 and fpm_dfe_elt==0):
                
        atmoNoCoro = data3[0,:]*data[0,:]
            
    # plt.plot(lam_lst*1e9, data[0,:])
    # plt.plot(lam_lst*1e9, data[0,:], label=f'{ncpa}', color=c_tab[i])
    # plt.plot(lam_lst*1e9, data[0,:], label=f'{disp}', color=c_tab[i])
    # plt.plot(lam_lst*1e9, data[0,:], label=f'{ls_ape:.1f}°', color=c_tab[i])
    # plt.plot(lam_lst*1e9, data[0,:], label=f'{offset*lamC*mas2rad/D:.1f}', color=c_tab[i])
    # plt.plot(lam_lst*1e9, data[0,:], label=f'{ls_voe*100./nPup:.2f}', color=c_tab[i])
    plt.plot(lam_lst*1e9, data[0,:], label=f'{ls_hoe*100./nPup:.2f}', color=c_tab[i])
    # plt.plot(lam_lst*1e9, data[0,:], label=f'{fpm_dfe_elt}', color=c_tab[i])
    # plt.plot(lam_lst*1e9, data[0,:], label=f'{offset*lamC*mas2rad/D:.1f} mas')
    
    plt.fill_between(lam_lst*1e9, data2[0,:], data2[1,:], alpha=0.2, color=c_tab[i])

bfiles = os.listdir((fdir_res2/'perfect'))
bfile = bfiles[np.argmax((lambda x:[len(i) for i in x])(bfiles))]
bdata = fits.getdata(fdir_res2/'perfect'/bfile)
   
plt.plot(lam_lst*1e9, bdata[:,pos_as_oi], color='black', ls=':')  #  , label='no atmo., coro.')
# plt.plot(lam_lst*1e9, atmoNoCoro,color='black' , label='atmo., no coro.', ls='--')

plt.ylim(1e-5,1e-1)
# plt.legend(title='NCPA [nm RMS]',fontsize='small', loc=2, ncol=2)
# plt.legend(title='Chromatic dispersion\n[mas/µm]',fontsize='small', loc=2, ncol=2)
# plt.legend(title='Lyot stop\nclocking error',fontsize='small', loc=2, ncol=1)
# plt.legend(title='Tilt [mas]',fontsize='small', loc=2, ncol=1)
# plt.legend(title=f'Lyot stop elevation offset [%D$_{{ELT}}]$\n(D$_{{LS}}$ = {diam} D$_{{ELT}}$)',fontsize='small', loc=2, ncol=2)
plt.legend(title=f'Lyot stop azimuth offset [%D$_{{ELT}}]$\n(D$_{{LS}}$ = {diam} D$_{{ELT}}$)',fontsize='small', loc=2, ncol=2)
# plt.legend(title='FPM defocus [nm RMS]',fontsize='small', loc=2, ncol=2)

plt.plot(lam_lst*1e9, atmoNoCoro,color='black' , ls='--')

#♠, label='atmo., no coro.')
bbox = dict(boxstyle='square', fc='w', alpha=0.125)
# plt.text(2000,2e-2,'--- atmo., no coro.', bbox=bbox)
# plt.text(2000,2e-5,'... no atmo., coro.', bbox=bbox)
plt.text(1600,2e-2,'--- atmo., no coro.', bbox=bbox)
plt.text(1600,2e-5,'... no atmo., coro.', bbox=bbox)


# fname_contrast_25mas = ('all_windshake_data_contrast_25mas_asRatioOf_lbd2D_ringAvgdPrfs_vs_wvl_' +
#                     os.path.basename(file_cro[0]).split('.')[0])
#if actual lambda/D
# fname_contrast_25mas = ('all_windshake_data_contrast_' + ptrn +'_'+as_str+'mas_'+ was_donow)
# fpath_contrast_25mas_svg = fdir_plt / (fname_contrast_25mas + '.svg')
# fpath_contrast_25mas_pdf = fdir_plt / (fname_contrast_25mas + '.pdf')
# plt.savefig(fpath_contrast_25mas_svg)
# plt.savefig(fpath_contrast_25mas_pdf)

# plt.savefig('C:/Users/asp/Desktop/tilt_'+jq+'.svg')
plt.savefig(fdir_plt / was_donow[-1] / ('ls_azimuthal_offset_'+str(int(as_oi))+'mas_'+str(int(spxl))+'mas_'+jq+'.pdf'), bbox_inches='tight', pad_inches=0.1)
# plt.savefig('C:/Users/asp/Desktop/tilt_'+jq+'.png')

plt.show()
    
    