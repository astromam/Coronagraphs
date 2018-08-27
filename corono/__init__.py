#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:59:00 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> (https://github.com/astromam)

License: MIT license

"""

from . import design, optim_1d, optim_2d
from . import utils

from .utils import reload, sft, isft, uniform_disk, radius_disk, sft_even
from .utils import to_dict, write_apod1d, load_apod1d, update_params

__version__ = '0.1.0b'

lst_submodule = ['design', 'optim_1d', 'optim_2d']
lst_utils = ['reload', 'sft', 'isft', 'sft_even', 'uniform_disk', 'radius_disk',
             'to_dict', 'write_apod1d', 'load_apod1d',
             'update_params']

__all__ = lst_submodule + lst_utils
