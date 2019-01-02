#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 17:47:09 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

#%% Initialization
import corono as coro

#%%
"""
Test the default params
"""
def test_get_default_params():
    
    coro.default.get_default_params_Coronagraph()
    
    coro.default.get_default_params_APLC1d()
    
    coro.default.get_default_params_SP1d()
    
    coro.default.get_default_params_DZPM1d()
    
    coro.default.get_default_params_HDZPM1d()

    coro.default.get_default_params_APLC2d()
    
    coro.default.get_default_params_SP2d()
    
    coro.default.get_default_params_DZPM2d()
