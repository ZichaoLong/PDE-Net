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
    errs.append(torch.load('checkpoint/'+name+'/errs'+str(r)))
    os.remove('checkpoint/'+name+'/errs'+str(r))
    updateerrs.append(torch.load('checkpoint/'+name+'/updateerrs'+str(r)))
    os.remove('checkpoint/'+name+'/updateerrs'+str(r))
    uvars.append(torch.load('checkpoint/'+name+'/uvars'+str(r)))
    os.remove('checkpoint/'+name+'/uvars'+str(r))
    updatevars.append(torch.load('checkpoint/'+name+'/updatevars'+str(r)))
    os.remove('checkpoint/'+name+'/updatevars'+str(r))
errs = np.concatenate(errs, axis=1)
updateerrs = np.concatenate(updateerrs, axis=1)
uvars = np.concatenate(uvars, axis=0)
updatevars = np.concatenate(updatevars, axis=0)
torch.save(errs, 'checkpoint/'+name+'/errs')
torch.save(updateerrs, 'checkpoint/'+name+'/updateerrs')
torch.save(uvars, 'checkpoint/'+name+'/uvars')
torch.save(updatevars, 'checkpoint/'+name+'/updatevars')
#%%
