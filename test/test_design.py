#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:10:20 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

#%% Initialization
import numpy as np
import corono as coro
import os
    
#%%
"""
Test the Coronagraph class
"""
def test_Coronagraph():
    
    corono0=coro.design.Coronagraph()
    
    # load and save params
    corono0.save_params('temp.file')
    corono0.load_params('temp.file')
    os.remove('temp.file')
    
    # print and filename
    print(corono0)
    print(corono0.get_filename())

    # print and filename
    varname = 'test'
    print(corono0.get_cache(varname))
    
    # test params
    assert 'nFPM' in corono0
    assert not 'mamadoupowa' in corono0
    
    # test values
    np.testing.assert_allclose(corono0.ClearPupil1d,1)
    
    # test function call
    assert 'Pupil1d' in corono0 
    corono0.compute_nostop_field_1d()
    
    corono0.compute_nostop_intensity_1d(poly=True)
    corono0.compute_nostop_intensity_1d(poly=False)
    
    
    corono0.generate_area()
    corono0.params['Pupil2dSym'] = True
    corono0.generate_area()
    
#    assert f.shape[1]==corono0.nPup and f.shape[0]==corono0.nlam

#    Apod = corono0.Pupil1d
#    corono0.compute_direct_intensity_1d(Apod, poly=True)

    
#%%    
def test_APLC1d():
    
    corono0=coro.design.APLC1d()
    
    Apod = corono0.Pupil1d
    corono0.compute_direct_field_1d(Apod)
    corono0.compute_corono_field_1d(Apod)
    
    corono0.compute_direct_intensity_1d(Apod, poly=True)
    corono0.compute_direct_intensity_1d(Apod, poly=False)
    corono0.compute_corono_intensity_1d(Apod, poly=True)
    corono0.compute_corono_intensity_1d(Apod, poly=False)
    

#%%    
def test_SP1d():
    
    corono0=coro.design.SP1d()
    
    Apod = corono0.Pupil1d
    corono0.compute_direct_field_1d(Apod)
    corono0.compute_corono_field_1d(Apod)
    
    corono0.compute_direct_intensity_1d(Apod, poly=True)
    corono0.compute_direct_intensity_1d(Apod, poly=False)
    corono0.compute_corono_intensity_1d(Apod, poly=True)
    corono0.compute_corono_intensity_1d(Apod, poly=False)
       
    
#%%
def test_DZPM1d():

    corono0=coro.design.DZPM1d()
    
    Apod = corono0.Pupil1d
    corono0.compute_direct_field_1d(Apod)
    corono0.compute_corono_field_1d(Apod)
    
    corono0.compute_direct_intensity_1d(Apod, poly=True)
    corono0.compute_direct_intensity_1d(Apod, poly=False)
    corono0.compute_corono_intensity_1d(Apod, poly=True)
    corono0.compute_corono_intensity_1d(Apod, poly=False)
    
#%%
def test_HDZPM1d():

    corono0=coro.design.HDZPM1d()
    
    Apod = corono0.Pupil1d
    corono0.compute_direct_field_1d(Apod)
    corono0.compute_corono_field_1d(Apod)
    
    corono0.compute_direct_intensity_1d(Apod, poly=True)
    corono0.compute_direct_intensity_1d(Apod, poly=False)
    corono0.compute_corono_intensity_1d(Apod, poly=True)
    corono0.compute_corono_intensity_1d(Apod, poly=False)
    

#%%
def test_HTZPM1d():

    corono0=coro.design.HTZPM1d()
    
    Apod = corono0.Pupil1d
    corono0.compute_direct_field_1d(Apod)
    corono0.compute_corono_field_1d(Apod)
    
    corono0.compute_direct_intensity_1d(Apod, poly=True)
    corono0.compute_direct_intensity_1d(Apod, poly=False)
    corono0.compute_corono_intensity_1d(Apod, poly=True)
    corono0.compute_corono_intensity_1d(Apod, poly=False)
    

#%%
def test_APLC2d():

    corono0=coro.design.APLC2d()
    
    Apod = corono0.Pupil2d
    corono0.compute_direct_field_2d(Apod)
    corono0.compute_corono_field_2d(Apod)
    corono0.compute_direct_lyot_field_2d(Apod)
    corono0.compute_corono_lyot_field_2d(Apod)
    
    corono0.compute_direct_intensity_2d(Apod, poly=True)
    corono0.compute_direct_intensity_2d(Apod, poly=False)
    corono0.compute_corono_intensity_2d(Apod, poly=True)
    corono0.compute_corono_intensity_2d(Apod, poly=False)
    
    corono0.compute_direct_field_2d_vec(Apod)
    corono0.compute_corono_field_2d_vec(Apod)
    
    params = coro.to_dict() 
    params2 = coro.update_params(params, Pupil2dSym=True)
    corono0=coro.design.APLC2d(**params2)
    corono0.compute_direct_field_2d(Apod)
    corono0.compute_corono_field_2d(Apod)
    corono0.compute_direct_lyot_field_2d(Apod)
    corono0.compute_corono_lyot_field_2d(Apod)
    
    OPDmap2d = corono0.Pupil2d
    params3 = coro.update_params(params, OPDmap2d=OPDmap2d)
    corono0=coro.design.APLC2d(**params3)
    corono0.compute_direct_field_2d(Apod)
    corono0.compute_corono_field_2d(Apod)
    corono0.compute_direct_lyot_field_2d(Apod)
    corono0.compute_corono_lyot_field_2d(Apod)

    Ampmap2d = corono0.Pupil2d    
    params4 = coro.update_params(params, Ampmap2d=Ampmap2d)
    corono0=coro.design.APLC2d(**params4)
    corono0.compute_direct_field_2d(Apod)
    corono0.compute_corono_field_2d(Apod)
    corono0.compute_direct_lyot_field_2d(Apod)
    corono0.compute_corono_lyot_field_2d(Apod)
    
#%%
def test_SP2d():

    corono0=coro.design.SP2d()
    
    Apod = corono0.Pupil2d
    corono0.compute_direct_field_2d(Apod)
    corono0.compute_corono_field_2d(Apod)

    corono0.compute_direct_intensity_2d(Apod, poly=True)
    corono0.compute_direct_intensity_2d(Apod, poly=False)
    corono0.compute_corono_intensity_2d(Apod, poly=True)
    corono0.compute_corono_intensity_2d(Apod, poly=False)
    
    corono0.compute_direct_field_2d_vec(Apod)
    corono0.compute_corono_field_2d_vec(Apod)
    
    params = coro.to_dict() 
    params2 = coro.update_params(params, Pupil2dSym=True)
    corono0=coro.design.SP2d(**params2)
    corono0.compute_direct_field_2d(Apod)
    corono0.compute_corono_field_2d(Apod)

 #%%
def test_DZPM2d():

    corono0=coro.design.DZPM2d()
    
    Apod = corono0.Pupil2d
    corono0.compute_direct_field_2d(Apod)
    corono0.compute_corono_field_2d(Apod)
    
    corono0.compute_direct_intensity_2d(Apod, poly=True)
    corono0.compute_direct_intensity_2d(Apod, poly=False)
    corono0.compute_corono_intensity_2d(Apod, poly=True)
    corono0.compute_corono_intensity_2d(Apod, poly=False)
    
    corono0.compute_direct_field_2d_vec(Apod)
    corono0.compute_corono_field_2d_vec(Apod)

    params = coro.to_dict() 
    params2 = coro.update_params(params, Pupil2dSym=True)
    corono0=coro.design.DZPM2d(**params2)
    corono0.compute_direct_field_2d(Apod)
    corono0.compute_corono_field_2d(Apod)    