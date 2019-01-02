import numpy as np

'''
Simple script to test memory allocation on telserv3 & 2
Array size causing problems:

npup = 486 x 486
5 - 10 lam/D
0% BW

npp = 
nvv =

A2tmp.shape = 
AZ0vv.shape = 
A2.shape =

self.A.shape =

'''

npp    = 
nvv    = 

A2tmp  = -np.identity(npp)
AZ0vv  =  np.zeros((self.nvv, self.npp))

A2     = np.concatenate((A2tmp,AZ0vv))

test_A = np.random.rand(,)

test_A = np.concatenate((self.A,A2,-A2), axis = 1)