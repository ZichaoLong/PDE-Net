import sys
import time
import numpy as np
import torch
import conf,setenv

repeatnum = 20
issparse = sys.argv[1]
block = sys.argv[2]
terms0 = []
terms1 = []
coeffs0 = np.zeros((repeatnum,40))
coeffs1 = np.zeros((repeatnum,40))
for testnum in range(20):
    D = torch.load('coeffs/'+str(testnum)+'-'+str(issparse)+'-'+str(block))
    terms0.append(D['terms0'])
    terms1.append(D['terms1'])
    coeffs0[testnum,:] = D['coeffs0']
    coeffs1[testnum,:] = D['coeffs1']
torch.save(dict(terms0=terms0,terms1=terms1,coeffs0=coeffs0,coeffs1=coeffs1),'coeffs/'+str(issparse)+'-'+str(block))
