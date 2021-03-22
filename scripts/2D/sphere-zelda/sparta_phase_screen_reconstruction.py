import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.time import Time
from pathlib import Path

import vigan.optics.aperture as aperture
import vigan.ao as ao
import vigan.optics.zernike as zernike

import sys
import pwd
import os

#%%
#
# SPARTA reconstruction parameters
#
t_max             = 30    # length of the required sequence, in seconds
reconstruct_turb  = True
include_tiptilt   = True
include_gains     = False
plot_stats        = False
save_intermediate = False

#
# AO simulation
#
add_fitting      = True
add_aliasing     = True

seeing  = 0.7
L0      = 25
z       = [  0,   4, 16]   # [  0,  4, 16]
Cn2     = [ 55,  35, 10]   # [ 55, 35, 10]
v       = [  8,  10, 36]   # [  8, 10, 20]
arg_v   = [  0, -45,  0]
mag     = 3.29
zenith  = 90-74
azimuth = 71
spaf    = 0.5
wave    = 1.593e-6
seed    = 12345

#%%
user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

if user == 'mndiaye':
    if syst == 'darwin':
        fdir = Path('~/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs').expanduser()
        path_root = Path('~/OneDrive - Université Nice Sophia Antipolis/data/zelda/ZELDA-2018/SPARTA/').expanduser()
        path_sparta_config = Path('~/OneDrive - Université Nice Sophia Antipolis/data/zelda/ZELDA-2018/SAXO/').expanduser()
    elif syst == 'linux':
        fdir = Path('/scratch/{0}/data/Coronagraphs/'.format(user)).resolve()
        path_root = Path('/scratch/{0}/data/zelda/ZELDA-2018/SPARTA/'.format(user)).expanduser()
        path_sparta_config = Path('/scratch/{0}/data/zelda/ZELDA-2018/SAXO/'.format(user)).expanduser()        
    else:
        raise ValueError('Unknown operating system {0}'.format(user))
else:
    raise ValueError('Unknown user {0}'.format(user))

#%%
#
# dates
#
#path_root = Path('~/data/ZELDA/2018-04-03_night/SPARTA/').expanduser()

date = '2018-04-04T00-41-34'  # ZELDA test
#date = '2018-04-04T00:42:51'  # ZELDA test
#date = '2018-04-04T03:06:15'  # ZELDA/COFFEE test - average conditions
#date = '2018-04-04T03:12:50'  # ZELDA/COFFEE test - bad conditions

# path_root = Path('~/data/ZELDA/2018-06-27/SPARTA/').expanduser()
# date = '2018-06-27T05_04_28'
#%%
#
# static SAXO configurations
#
#path_sparta_config = Path('~/data/ZELDA/SAXO/').expanduser()

dim_pup = 240

# influence matrix normalization = defoc optique en rad @ 632 nm
IMF = fits.getdata(path_sparta_config / 'SAXO_DM_IFM.fits')
rad_632_to_nm_opt = 632 / (2*np.pi)
IMF = (IMF * rad_632_to_nm_opt).reshape(1377, 240*240).T
#%%
#
# read data
#
path_VisLoopRecorder = path_root / '{0}-VisLoopRecorder'.format(date)

VisLoopRecorder = fits.getdata(path_VisLoopRecorder / '{0}-VisLoopRecorder.fits'.format(date))
RefMap          = fits.getdata(path_VisLoopRecorder / 'VisHOCtr-ACT_POS_REF_MAP.fits').squeeze()
ITT_IM          = fits.getdata(path_VisLoopRecorder / 'CLMatrixOptimiser-ITT_IM.fits')   # IM == V2S
S2M             = fits.getdata(path_VisLoopRecorder / 'CLMatrixOptimiser-S2M.fits')
M2V             = fits.getdata(path_VisLoopRecorder / 'CLMatrixOptimiser-M2V.fits')
V2M             = fits.getdata(path_VisLoopRecorder / 'CLMatrixOptimiser-V2M.fits')
M2S             = fits.getdata(path_VisLoopRecorder / 'CLMatrixOptimiser-M2S.fits')
PROJ_ORTH       = fits.getdata(path_VisLoopRecorder / 'CLMatrixOptimiser-PROJ_ORTH.fits')
PROJ_PAR        = fits.getdata(path_VisLoopRecorder / 'CLMatrixOptimiser-PROJ_PAR.fits')

# make matrices from arrays
ITT_IM    = np.matrix(ITT_IM)
S2M       = np.matrix(S2M)
M2V       = np.matrix(M2V)
V2M       = np.matrix(V2M)
M2S       = np.matrix(M2S)
PROJ_ORTH = np.matrix(PROJ_ORTH)
PROJ_PAR  = np.matrix(PROJ_PAR)

HODM_Positions  = np.matrix(VisLoopRecorder['HODM_Positions'])
Gradients       = np.matrix(VisLoopRecorder['Gradients'])
TimeStamps      = VisLoopRecorder['Seconds']
#%%
#
# useful parameters
#

# create tip-tilt zernike modes
pupil = aperture.disc_obstructed(dim_pup, dim_pup, 0.14, diameter=True, cpix=False)
zern_tip, zern_tilt = np.meshgrid(np.linspace(-0.5, 0.5, dim_pup), np.linspace(-0.5, 0.5, dim_pup))

zern_tip  *= pupil
zern_tilt *= pupil

# timing of the data
sec = Time(TimeStamps, format='unix')

time_start    = sec[0]
time_end      = sec[-1]
ellapsed_time = time_end - time_start
delta_t       = ellapsed_time.sec/len(sec)
freq          = 1/delta_t

print('Ellapsed time: {0:5.1f}s'.format(ellapsed_time.sec))
print('Frequency of {0:6.1f} Hz'.format(freq))

time_ellapsed_array = np.arange(len(sec))*delta_t

# index in frames
i_max = np.where(time_ellapsed_array <= t_max)[0].max()

# extract data
n_screen = i_max
HODM_Positions = HODM_Positions[0:n_screen, ...]
Gradients      = Gradients[0:n_screen, ...]
#%%
#
# tip-tilt slope-to-volt matrix
#

ITT_IM_par = PROJ_PAR @ ITT_IM          # V2Spar
ITT_Spar2V = np.linalg.inv(ITT_IM_par)  # Spar2V
#%%
#
# turbulence
#
if reconstruct_turb:
    HODM_Positions_NoBias = HODM_Positions - RefMap
    
    print('Turbulence reconstruction')
    volt = HODM_Positions_NoBias[0:i_max]
    turbulence = (volt @ IMF.T).reshape((i_max, 240, 240))
    del volt
    
    print(' ==> save')
    fits.writeto(path_VisLoopRecorder / '{0}-saxo_turbulence.fits'.format(date), turbulence, overwrite=True)
    del turbulence
#%%
#
# residual turbulence
#
print('Residual turbulence reconstruction')

slope_orth = Gradients @ PROJ_ORTH.T
slope_par  = Gradients @ PROJ_PAR.T

# ==> high order
print(' ==> high-orders')
mode = slope_orth @ S2M.T
del slope_orth
volt = mode @ M2V.T
del mode
res_turbulence = np.array(volt @ IMF.T)
del volt

# more direct computation:
# slope_orth = PROJ_ORTH @ Gradients.T
# slope_par  = PROJ_PAR  @ Gradients.T
# mode = S2M @ slope_orth
# volt = M2V @ mode
# res_turbulence = np.array(IMF @ volt)

if save_intermediate:
    tmp = np.reshape(res_turbulence, (i_max, 240, 240))
    fits.writeto(path_VisLoopRecorder / 'turbulence_ho.fits', tmp, overwrite=True)
    del tmp

if include_gains:
    res_turbulence *= 0.8

# ==> tip-tilt
if include_tiptilt:
    print(' ==> tip-tilt')
    volt = np.array(slope_par @ ITT_Spar2V.T)
    del slope_par
    tt_nm_pv = 8e9 * np.tan(volt * 2.6 / 3600 * np.pi/180)

    tt_nm_std = tt_nm_pv.std(axis=0)
    jitter = np.sqrt(np.sum(((1642e-9/8*180/np.pi*3600*1000) / (1642/tt_nm_std))**2))
    print('Jitter = {:.1f} mas'.format(jitter))
    
    # ==> tip
    tip_lin = (zern_tip.ravel())[np.newaxis]
    res_tip = (tip_lin*tt_nm_pv[:, 0][..., np.newaxis])
    res_turbulence += res_tip
    if save_intermediate:
        tmp = np.reshape(res_tip, (i_max, 240, 240))
        fits.writeto(path_VisLoopRecorder / 'turbulence_tip.fits', tmp, overwrite=True)
        del tmp
    del res_tip
    print('  * tip added')

    # ==> tilt
    tilt_lin = (zern_tilt.ravel())[np.newaxis]
    res_tilt = (tilt_lin*tt_nm_pv[:, 1][..., np.newaxis])
    res_turbulence += res_tilt
    if save_intermediate:
        tmp = np.reshape(res_tilt, (i_max, 240, 240))
        fits.writeto(path_VisLoopRecorder / 'turbulence_tilt.fits', tmp, overwrite=True)
        del tmp
    del res_tilt
    print('  * tilt added')
#%%
# reshape
print(' ==> reshape')
res_turbulence = res_turbulence.reshape((i_max, 240, 240))
res_turbulence = np.rot90(res_turbulence, -1, axes=(1, 2))
#%%
# fitting + aliasing
if add_fitting or add_aliasing:
    print(' ==> simulated fitting and aliasing errors')
    phs = ao.residual_screen_sphere(seeing, L0, z, Cn2, v, arg_v, mag, zenith, azimuth, 
                                    spat_filter=spaf, img_wave=wave, dim_pup=dim_pup,
                                    n_screen=n_screen, chunk_size=500,
                                    fit=add_fitting, servo=False, alias=add_aliasing, noise=False,
                                    diff_refr=False, psd_only=False, seed=seed)
    phs = phs[..., dim_pup:2*dim_pup, dim_pup:2*dim_pup]
    phs *= pupil
    res_turbulence += phs*1e9
    if save_intermediate:
        tmp = np.reshape(phs*1e9, (i_max, 240, 240))
        fits.writeto(path_VisLoopRecorder / 'turbulence_fitting+aliasing.fits', tmp, overwrite=True)
        del tmp    
    del phs
#%%
# save
fname = '{}-saxo_residual_turbulence_time={:04.1f}sec_seeing={:.1f}as_tiptilt={}_gains={}_fitting={}_alias={}.fits'.format(date, t_max, seeing, int(include_tiptilt), int(include_gains), int(add_fitting), int(add_aliasing))
print(' ==> save to {}'.format(fname))
fits.writeto(path_VisLoopRecorder / fname, res_turbulence, overwrite=True)
# fits.writeto(Path('/Users/avigan/data/ZELDA/Simulations/') / fname, res_turbulence, overwrite=True)

#%%
# plot statistics
# if plot_stats:
#     ipup = np.where(res_turbulence[0].ravel() != 0)[0]
#     tmp = res_turbulence.reshape((i_max, -1))
#     tmp = tmp[:, ipup]
#     std = tmp.std(axis=1)

#     plt.plot(std)
#     plt.axhline(std.mean(), color='r')
#     plt.show()
#     print(std.mean())
#%%    
# free memory
del res_turbulence


