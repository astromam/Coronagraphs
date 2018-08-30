#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 22:49:45 2018

Author: Mamadou N'Diaye <mamadou.ndiaye@oca.eu> 

License: MIT license

"""

import corono as coro
import os

#%%
"""
Function to check basics in the class and subclass
"""

#%%
"""
Test the ProblemMatrix class
"""
def test_ProblemMatrix():
    
    problem0 = coro.optim_1d.ProblemMatrix()
    
    problem1 = coro.optim_1d.ProblemMatrix(corono=coro.design.APLC1d())
    
    assert 'cDarkHole' in problem0

    print(problem0.params['cDarkHole'])    
    print(problem0.params)
    
    # load and save params
    problem0.save_params('temp.file')
    problem0.load_params('temp.file')
    os.remove('temp.file')
    
    assert 'problem_name' in problem0
    problem0.compute_response_matrices()

#    assert 'corono_name' in problem0
#    problem0.get_filename()

#    problem0.compute_problem_matrices()
#
#    problem0.solve_model()

    
    problem0.print_log('test')
    