import sys
import time
import numpy as np
import torch
import conf,setenv

dataname = sys.argv[1]
basename = sys.argv[2]
repeatnum = int(sys.argv[3])
block = sys.argv[4]
terms0 = []
terms1 = []
coeffs0 = np.zeros((repeatnum,40))
coeffs1 = np.zeros((repeatnum,40))
for testnum in range(repeatnum):
    D = torch.load('coeffs/'+dataname+'/'+basename+'-id'+str(testnum)+'-block'+str(block))
    terms0.append(D['terms0'])
    terms1.append(D['terms1'])
    coeffs0[testnum,:] = D['coeffs0']
    coeffs1[testnum,:] = D['coeffs1']
torch.save(dict(terms0=terms0,terms1=terms1,coeffs0=coeffs0,coeffs1=coeffs1),'coeffs/'+dataname+'/'+basename+'-block'+str(block))
