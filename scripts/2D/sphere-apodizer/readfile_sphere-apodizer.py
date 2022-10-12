#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 09:45:21 2022

@author: mndiaye
"""

#%%
"""
### Initialization
"""
import os
import pwd
import sys

user = pwd.getpwuid(os.getuid())[0]
syst = sys.platform

import numpy as np
import pandas as pd
from pathlib import Path

#%%
"""
### Directory
"""
# if user == 'mndiaye':
#     if syst == 'darwin':
#         fdir = Path('/Users/mndiaye/OneDrive - Université Nice Sophia Antipolis/data/Coronagraphs/data/2D/pupils/').resolve()
#     elif syst == 'linux':
#         fdir = Path('/scratch/mndiaye/data/Coronagraphs/data/2D/pupils/').resolve()
#     else:
#         raise ValueError('Unknown operating system {0}'.format(syst))
# else:
#     raise ValueError('Unknown user {0}'.format(user))


fdir_apod = Path('/Users/mndiaye/Nextcloud/SPHERE_upgrade/WP_corono/2009_apodizers').resolve()

fname = '2009.09.25_IR_Apodizer_DesignA_profile_v01.xls'
fpath = fdir_apod / fname

#%%
"""
### Read file with pandas
"""
apod_file = pd.read_excel(fpath)

apod_file.info()