# -*- coding: utf-8 -*-
"""
Created on Mon may  9 2025

@author: asp
"""

# plot contrats at given angular separation for two contrast profile
# fits files produced by multi_wvl_psf_profile_cut.py

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
# import os
from pathlib import Path
# from psf_profile import profile
from datetime import datetime  #  asp for datetime of now

#fontsize to 15 for all plots
plt.rcParams.update({'font.size': 14})  #♦  mdiaye 15!


#%%
"""
### scaling
"""
# lam_c = 1600e-9  #  "central" lambda (of interest) in meters
# conversion lradian to mas
rad2mas = np.pi/(180.*3600*1000)
mas2rad = 1/rad2mas

as_oi = 20
as_str = str(int(as_oi))
rW_mas = 7


#%%
"""
### Working directories
"""
user = 'Alain'
if user == 'Alain':
    fdir_res   = Path('D:/Andes/Data_corono/results/').resolve()  #  fits data
    fdir_plt   = Path('D:/Andes/Data_corono/plots/').resolve()   #  plots
    fdir_dat = Path("D:/Andes/Data_corono/data/OPDs_PASSATA/OPD/WS/").resolve()

donow = datetime.now().strftime("%Y%m%d%H%M%S")  #  asp, datetime of now
   
res_dir = fdir_res
# fdir_plt = fdir_plt / donow
# os.makedirs(fdir_plt, exist_ok=True)

#%%
# 500 Hz    0.9/0.37/4.0    20250604120819 , 10 sets, lam_itv = 80 nm
# 1 kH0 Hz  0.9/0.37/4.0    20250602181544 , 10 sets, lam_itv = 80 nm
# 500 Hz    0.9/0.37/4.0    20250610135805 , 10 sets, lam_itv = 50 nm
# 1 kH0 Hz  0.9/0.37/4.0    20250610140016 , 10 sets, lam_itv = 50 nm

ptrn = str(int(as_oi))+"mas_"+str(int(rW_mas))+"mas_"


#%%
# 500Hz  0.9/0.37/4.0    20250602181544 , 10 sets constant WS

# files = ("20250604120819/contrast_"+ptrn+"JQM_20250604120819.fits",
#         "20250602181544/contrast_"+ptrn+"JQM_20250602181544.fits",
#         "20250602181544/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250522160547_"+ptrn+"960_1840nm.fits",
#         "20250602181544/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250522160547_"+ptrn+"960_1840nm.fits")

# ?
# files = ("20250610135805/contrast_"+ptrn+"JQM_20250610135805.fits",
#         "20250610140016/contrast_"+ptrn+"JQM_20250610140016.fits",
#         "20250610140016/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250611134316_"+ptrn+"950_1850nm.fits",
#         "20250610140016/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250611134316_"+ptrn+"950_1850nm.fits")

# 20250711155826 0.89/0.35/3.9 1.kHz
# 20250711155801 0.89/0.35/3.9 .5kHz
# files = ("20250711155801/contrast_"+ptrn+"JQM_20250711155801.fits",
#         "20250711155826/contrast_"+ptrn+"JQM_20250711155826.fits",
#         "20250711155826/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250711155826_"+ptrn+"950_1850nm.fits",
#         "20250711155826/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250711155826_"+ptrn+"950_1850nm.fits")

# 20250711155932 0.88/0.30/4.0 1.kHz
# 20250711155911 0.88/0.30/4.0 .5kHz
# files = ("20250711155911/contrast_"+ptrn+"JQM_20250711155911.fits",
#         "20250711155932/contrast_"+ptrn+"JQM_20250711155932.fits",
#         "20250711155932/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250711155932_"+ptrn+"950_1850nm.fits",
#         "20250711155932/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250711155932_"+ptrn+"950_1850nm.fits")

#%%
# cameras comparison
# 2 cameras
# 0.89 0.35 3.9 alice  20251007190203
# 0.89 0.35 3.9 ocam2k 20251007190218
# files = ("20251007190203/contrast_"+ptrn+"JQM_20251007190203.fits",
#         "20251007190218/contrast_"+ptrn+"JQM_20251007190218.fits",
#         "20251007190218/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20251007190218_"+ptrn+"950_1850nm.fits",
#         "20251007190218/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20251007190218_"+ptrn+"950_1850nm.fits")

# 0.90 0.37 4.0 alice  20251007190322
# 0.90 0.37 4.0 ocam2k 20251007190259
# files = ("20251007190322/contrast_"+ptrn+"JQM_20251007190322.fits",
#         "20251007190259/contrast_"+ptrn+"JQM_20251007190259.fits",
#         "20251007190259/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20251007190259_"+ptrn+"950_1850nm.fits",
#         "20251007190259/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20251007190259_"+ptrn+"950_1850nm.fits")

# 3 cameras
# 0.9 0.37 4.0  cam_500us 20251013105414
# 0.90 0.37 4.0 alice  20251007190322
# 0.90 0.37 4.0 ocam2k 20251007190259
# files = ("20251007190322/contrast_"+ptrn+"JQM_20251007190322.fits",
#         "20251007190259/contrast_"+ptrn+"JQM_20251007190259.fits",
#         "20251007190259/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20251007190259_"+ptrn+"950_1850nm.fits",
#         "20251007190259/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20251007190259_"+ptrn+"950_1850nm.fits",
#         "20251013105414/contrast_"+ptrn+"JQM_20251013105414.fits",
#         "20251013105414/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20251013105414_"+ptrn+"950_1850nm.fits",
#         "20251013105414/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20251013105414_"+ptrn+"950_1850nm.fits")

# 0.89 0.35 3.9 cam_500us 20251013105528
# 0.89 0.35 3.9 alice  20251007190203
# 0.89 0.35 3.9 ocam2k 20251007190218
files = ("20251007190203/contrast_"+ptrn+"JQM_20251007190203.fits",
        "20251007190218/contrast_"+ptrn+"JQM_20251007190218.fits",
        "20251007190218/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20251007190218_"+ptrn+"950_1850nm.fits",
        "20251007190218/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20251007190218_"+ptrn+"950_1850nm.fits",
        "20251013105528/contrast_"+ptrn+"JQM_20251013105528.fits",
        "20251013105528/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20251013105528_"+ptrn+"950_1850nm.fits",
        "20251013105528/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20251013105528_"+ptrn+"950_1850nm.fits")

# 20250711160041 0.88/0.30/3.5 1.kHz
# 20250711160019 0.88/0.30/3.5 .5kHz
# files = ("20250711160019/contrast_"+ptrn+"JQM_20250711160019.fits",
#         "20250711160041/contrast_"+ptrn+"JQM_20250711160041.fits",
#         "20250711160041/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250711160041_"+ptrn+"950_1850nm.fits",
#         "20250711160041/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250711160041_"+ptrn+"950_1850nm.fits")

#%%
# 500Hz 0.86 0.31 4.0 20250709155129
#  1kHz 0.86 0.31 4.0 20250709155051
# files = ("20250709155129/contrast_"+ptrn+"JQM_20250709155129.fits",
#         "20250709155051/contrast_"+ptrn+"JQM_20250709155051.fits",
#         "20250709155051/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250709155051_"+ptrn+"950_1850nm.fits",
#         "20250709155051/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250709155051_"+ptrn+"950_1850nm.fits")

# 500Hz  0.9/0.37/4.0    20250708162615 , 10 sets
# files = ("20250708162615/contrast_"+ptrn+"JQM_20250708162615.fits",
#         "20250708171059/contrast_"+ptrn+"JQM_20250708171059.fits",
#         "20250708171059/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250708171059_"+ptrn+"950_1850nm.fits",
#         "20250708171059/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250708171059_"+ptrn+"950_1850nm.fits")

# 1 kH0 Hz  0.9/0.37/4.0    20250602181544 , 10 sets

# YJH 0.9 / 0.37 / 4.
# files = ("20250522161347/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250522161347.fits",
#         "20250522160547/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250522160547.fits",
#         "20250522160547/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250522160547.fits",
#         "20250522160547/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250522160547.fits")

# RIZ 0.9 / 0.37 / 4.
# files = ("20250521172353/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250521172353.fits",
#         "20250521172457/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250521172457.fits",
#         "20250521172457/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250521172457.fits",
#         "20250521172457/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250521172457.fits")

# 0.804 	 0.000341188 	 [0.92 0.36 3.9 ]    20250523162033 20250523162130
# files = ("20250523162033/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250523162033_959_1760.fits",
#         "20250523162130/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250523162130_959_1760.fits",
#         "20250523162130/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250523162130_959_1760.fits",
#         "20250523162130/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250523162130_959_1760.fits")

# 0.811 	 0.000354267 	 [0.92 0.35 3.8 ]    20250523162217 20250523162247
# files = ("20250523162217/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250523162217_959_1760.fits",
#         "20250523162247/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250523162247_959_1760.fits",
#         "20250523162247/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250523162247_959_1760.fits",
#         "20250523162247/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250523162247_959_1760.fits")

# 0.824 	 0.000380266 	 [0.93 0.36 3.6 ]    20250523163019 20250523162955 
# files = ("20250523163019/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250523163019_959_1760.fits",
#         "20250523162955/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250523162955_959_1760.fits",
#         "20250523162955/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250523162955_959_1760.fits",
#         "20250523162955/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250523162955_959_1760.fits")

# 0.826 	 0.000388391 	 [0.92 0.33 3.5 ]    20250523163116 20250523163139
# files = ("20250523163116/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250523163116_959_1760.fits",
#         "20250523163139/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250523163139_959_1760.fits",
#         "20250523163139/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250523163139_959_1760.fits",
#         "20250523163139/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250523163139_959_1760.fits")

# # 0.82 / 0.41 / 3.8
# files = ("20250521103343/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250521103343.fits",
#         "20250521135258/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250521135258.fits",
#         "20250521135258/perfect/contrast_profile_L0toD_perfect_wonoise_psf_20250521135258.fits",
#         "20250521135258/perfect/contrast_profile_L0toD_perfect_wonoise_coro_psf_20250521135258.fits")

# 0.82 / 0.37 / 3.0
# files = ("20250521085239/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250521085239.fits",
#         "20250521085213/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250521085213.fits",)

# 0.9 / 0.37 / 3.5
# files = ("20250520132136/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250520132136.fits",
#         "20250520132159/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250520132159.fits",)

# 0.96 / 0.3 /3.5 
# files = ("20250520154454/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250520154454.fits",
#         "20250520154438/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250520154438.fits",)

# 0.84 / 0.38 / 3.2
# files = ("20250520110756/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250520110756.fits",
#         "20250520110739/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250520110739.fits",)

# 0.96 / 0.3 / 3. 
# files = ("20250516134048/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250516134048.fits",
#         "20250516141104/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250516141104.fits",)

# 0.9 / 0.37 / 3.
# files = ("20250516165000/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250516165000.fits",
#         "20250516154501/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250516154501.fits",)

# 0.94 / 0.31 / 4.
# files = ("20250519111848/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250519111848.fits",
#         "20250519111554/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250519111554.fits")

# 0.92 / 0.33 / 4.
# files = ("20250519165120/20250508_163051/contrast_profile_L0toD_20250508_163051_ao_corr_coro_psf_20250519165120.fits",
#         "20250519165144/20250325_145800/contrast_profile_L0toD_20250325_145800_ao_corr_coro_psf_20250519165144.fits")

#%%

donow = files[0].split('/')[0]
fdir_plt = fdir_plt / donow

head_psf = fits.getheader(res_dir / files[0])

lamC = head_psf['LMBD']
lam_min = head_psf['LMIN']
lam_stp = head_psf['LSTP']
lam_itv = head_psf['LITV']
lam_lst = np.arange(lam_min,lam_min+(lam_itv)*lam_stp+1e-9,lam_stp)
lam_lst = lam_lst[np.where(lam_lst < 1860e-9)]
nL = len(lam_lst)

nImg = head_psf['NIMG']
D = head_psf['DIAM']
mB = np.round(head_psf['SFPM'],2)
dL = np.round(head_psf['FDIA'],2)
obs = np.round(head_psf['OBST'],2)
pscale = head_psf['PSCL']
hlf_fov = nImg * pscale / 2.

print('Lyot diam.:',dL,', Lyot obsc.:', obs, ', FPM width:', mB, 
      ', lambda ref.:', lamC)
   
aS = np.arange(nImg//2)*(mas2rad * hlf_fov / (D *1e9) ) # ang. sep.
pos_as_oi = int(np.median(np.argmin(np.abs(aS[:]-as_oi))))

nncr_data  = (fits.getdata(res_dir / files[3]))[0:nL,:]
contrast_nncr =  nncr_data[:,pos_as_oi]

atmo_data =  (fits.getdata(res_dir / files[2]))[0:nL,:]
atmo = atmo_data[:,pos_as_oi]

contrast = np.zeros((nL, 2))
# cont_std = np.zeros((nL, 2))
cont_min = np.zeros((nL, 2))
cont_max = np.zeros((nL, 2))
# maxi = np.zeros((nL, 2))

for i in range(2):

    # if contrast files from individual sets of opds
    # Int_D_prf_avg  = fits.getdata(res_dir / files[i]) # [0:nL,:]
    # Int_D_prf_min  = fits.getdata(res_dir / files[i], ext=2) # [0:nL,:]
    # Int_D_prf_max  = fits.getdata(res_dir / files[i], ext=3) # [0:nL,:]
    # contrast[:,i] = Int_D_prf_avg[:,pos_as_oi]
    # cont_min[:,i] = Int_D_prf_min[:,pos_as_oi]
    # cont_max[:,i] = Int_D_prf_max[:,pos_as_oi]

    # if contrast files from average of sets of opds
    contrast[:,i] = fits.getdata(res_dir / files[i])[0,:]
    cont_min[:,i] = fits.getdata(res_dir / files[i], ext=1)[0,:]
    cont_max[:,i] = fits.getdata(res_dir / files[i], ext=1)[1,:]

if len(files)==7:
    
    contrast2 = np.zeros((nL))
    # cont_std2 = np.zeros((nL, 2))
    cont_min2 = np.zeros((nL))
    cont_max2 = np.zeros((nL))

    # Int_D_prf_avg2  = fits.getdata(res_dir / files[4+i]) # [0:nL,:]
    # if contrast files from individual sets of opds
    # Int_D_prf_min2  = fits.getdata(res_dir / files[4+i], ext=2) # [0:nL,:]
    # Int_D_prf_max2  = fits.getdata(res_dir / files[4+i], ext=3) # [0:nL,:]
    # contrast[:,i] = Int_D_prf_avg2[:,pos_as_oi]
    # cont_min[:,i] = Int_D_prf_min2[:,pos_as_oi]
    # cont_max[:,i] = Int_D_prf_max2[:,pos_as_oi]

    # if contrast files from average of sets of opds
    contrast2 = fits.getdata(res_dir / files[4])[0,:]
    cont_min2 = fits.getdata(res_dir / files[4], ext=1)[0,:]
    cont_max2 = fits.getdata(res_dir / files[4], ext=1)[1,:]

        

            
#%%
"""
plot profiles
"""
# lbl = ["500Hz", "1kHz"]
# lbl = ["alice","ocam2k"]
lbl = ["alice","ocam2k","cam_500us"]
lst = ["-", "-","--"] 
clt = ["C0", "C1", "C4"]
# plot of the azimutal average ratio profile
plt.figure(2, (8, 4.5))
plt.tight_layout()
plt.xlabel(r'Wavelength $\lambda$ [nm]')#[$\lambda$/D]')
plt.ylabel(f'Contrast @ {int(as_oi)} mas')
plt.yscale('log')
# plt.title('Coronagraph configuration for YJH band')
plt.title(f'Photometric aperture {rW_mas} mas')
plt.grid(True)

for i in range(2):
    
    plt.plot(lam_lst*1e9, contrast[:,i], label=lbl[i], ls=lst[i], color=clt[i])
    plt.fill_between(lam_lst*1e9, cont_min[:,i], cont_max[:,i], alpha=0.2)

if len(files) == 7:
          
    plt.plot(lam_lst*1e9, contrast2, label=lbl[2],ls=lst[2], color=clt[2])
    plt.fill_between(lam_lst*1e9, cont_min2, cont_max2, alpha=0.2)
        
        
plt.plot(lam_lst*1e9, contrast_nncr, color='black', ls=':')
bbox = dict(boxstyle='square', fc='w', alpha=0.75)
# YJH
plt.text(930,3e-5,'.... no atmo., coro.', bbox=bbox)
# RIZ
# plt.text(650,2e-5,'... no atmo., coro.', bbox=bbox)

plt.plot(lam_lst*1e9, atmo, ls='--', color='black')
# YJH
plt.text(930,3e-2,'---- atmo., no coro.', bbox=bbox)
# RIZ
# plt.text(850,1e-2,'---- atmo., no coro.', bbox=bbox)
plt.xlim(900,1900)
plt.ylim(1e-5,1e-1)
# plt.legend(title='loop speed',fontsize='small', loc=4)
plt.legend(title='Camera:',fontsize='small', loc=4)

# fname = ("contrast_comp_"+as_str+'mas_'+str(int(dL*100))+'_'+str(int(obs*100))+
#          '_'+str(int(mB*10)))
# fname = ("contrast_comp_"+ptrn+
#          str(int(dL*100))+'_'+str(int(obs*100))+'_'+str(int(mB*10))+'_'+
#          str(int(np.rint(lam_lst[0]*1e9)))+'-'+
#          str(int(np.rint(lam_lst[-1]*1e9)))+'nm')
fname = ("contrast_comp_3cam_"+ptrn+
         str(int(dL*100))+'_'+str(int(obs*100))+'_'+str(int(mB*10))+'_'+
         str(int(np.rint(lam_lst[0]*1e9)))+'-'+
         str(int(np.rint(lam_lst[-1]*1e9)))+'nm')

# fpath_contrast_25mas_svg = fdir_plt / (fname + '.svg')
fpath_contrast_25mas_pdf = fdir_plt / (fname + '.pdf')
# fpath_contrast_25mas_png = fdir_plt / (fname + '.png')

# plt.savefig(fpath_contrast_25mas_svg)
plt.savefig(fpath_contrast_25mas_pdf, bbox_inches='tight', pad_inches=0.1)
# plt.savefig(fpath_contrast_25mas_png)

plt.show()

