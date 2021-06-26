#%%
import sys,os
import numpy as np
import torch

name = sys.argv[1]
if len(sys.argv)>2:
    repeatsize = int(sys.argv[2])
else:
    repeatsize = 1

errs = []
updateerrs = []
uvars = []
updatevars = []
for r in range(repeatsize):
    errs.append(torch.load('checkpoint/burgers-frozen-upwind-sparse0.005-noise0.001/errs'+str(r)))
    os.remove('checkpoint/burgers-frozen-upwind-sparse0.005-noise0.001/errs'+str(r))
    updateerrs.append(torch.load('checkpoint/burgers-frozen-upwind-sparse0.005-noise0.001/updateerrs'+str(r)))
    os.remove('checkpoint/burgers-frozen-upwind-sparse0.005-noise0.001/updateerrs'+str(r))
    uvars.append(torch.load('checkpoint/burgers-frozen-upwind-sparse0.005-noise0.001/uvars'+str(r)))
    os.remove('checkpoint/burgers-frozen-upwind-sparse0.005-noise0.001/uvars'+str(r))
    updatevars.append(torch.load('checkpoint/burgers-frozen-upwind-sparse0.005-noise0.001/updatevars'+str(r)))
    os.remove('checkpoint/burgers-frozen-upwind-sparse0.005-noise0.001/updatevars'+str(r))
errs = np.concatenate(errs, axis=1)
updateerrs = np.concatenate(updateerrs, axis=1)
uvars = np.concatenate(uvars, axis=0)
updatevars = np.concatenate(updatevars, axis=0)
torch.save(errs, 'checkpoint/burgers-frozen-upwind-sparse0.005-noise0.001/errs')
torch.save(updateerrs, 'checkpoint/burgers-frozen-upwind-sparse0.005-noise0.001/updateerrs')
torch.save(uvars, 'checkpoint/burgers-frozen-upwind-sparse0.005-noise0.001/uvars')
torch.save(updatevars, 'checkpoint/burgers-frozen-upwind-sparse0.005-noise0.001/updatevars')
#%%
