#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import numpy as np
import torch
import torchvision
import pdedata
import nonlinpdeconfig

configfile = 'checkpoint/'+'nonlinpde7x7moment2order-double'+'/options.yaml'

options = nonlinpdeconfig.setoptions(configfile=configfile,isload=True)

namestobeupdate, callback, nonlinpdelearner = nonlinpdeconfig.setenv(options)

globals().update(namestobeupdate)

callback.load(5)

#%% show learned coefficient
x = 2*np.pi*np.arange(200)/200
x,y = np.repeat(x[np.newaxis,:], 200, axis=0),np.repeat(x[:,np.newaxis], 200, axis=1)
x,y = torch.from_numpy(x),torch.from_numpy(y)
from torch.autograd import Variable
xy = Variable(torch.stack([x,y],dim=2))
nonlinpdelearner.xy = xy # set xy for pde-net
from pltutils import *
a = pltnewaxis(1,5)
k = 1
j = 0
for i in range(1,6):
    coe = eval('nonlinpdelearner.coe'+str(i))
    coe = coe().data.cpu().numpy()
    ax = a.flatten()[i-1]
    b = ax.imshow(coe, cmap='jet')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('C'+str(k-j)+str(j), fontsize=20)
    ax.figure.colorbar(b,ax=ax)
    j += 1
    if j > k:
        k += 1
        j = 0
nonlintermtest = nonlinpdelearner.xy.data.new(2000)
nonlintermtest.copy_(torch.linspace(nonlinear_interp_mesh_bound[0],nonlinear_interp_mesh_bound[1],2000))
testresults = nonlinpdelearner.nonlinearfitter(Variable(nonlintermtest))
testresults = testresults.data.cpu().numpy()
nonlintermtest = nonlintermtest.cpu().numpy()
a = pltnewaxis()
a.plot(nonlintermtest, testresults)
a.plot(nonlintermtest, np.sin(nonlintermtest)*nonlinear_coefficient)


#%% show prediction
trans = torchvision.transforms.Compose([pdedata.DownSample(5,'Dirichlet'), pdedata.ToTensor(), pdedata.ToPrecision(precision), pdedata.AddNoise(start_noise_level, end_noise_level)])
d = pdedata.singlenonlinear2d(np.array(range(1,teststepnum+1))*dt, mesh_size=mesh_size*5, initfreq=initfreq, diffusivity=diffusivity, nonlinear_coefficient=nonlinear_coefficient, transform=trans)
sample = pdedata.ToVariable()(d[0])
xy = torch.stack([sample['x'],sample['y']], dim=2)
u0 = sample['u0']
uT = sample['uT']
u0 = (u0.cuda(gpu) if gpu>=0 else u0)
nonlinpdelearner.id.MomentBank.moment.volatile = True
nonlinpdelearner.xy = xy # set xy for pde-net
F = pltnewmeshbar((3,5))
i = 0
for j in (0,20,40,60,80):
    if j == 0:
        u_true = u0.data.cpu().numpy()
        u = u_true
    else:
        u_true = uT[:,:,j-1].data.cpu().numpy()
        u = nonlinpdelearner(u0, j).data.cpu().numpy()
    err = u-u_true
    F(u_true,(0,i))
    F.a[0,i].set_title(r''+str(j)+' $\delta t$', fontsize=20)
    F.a[0,i].set_xticks([])
    F.a[0,i].set_yticks([])
    F(u, (1,i))
    F.a[1,i].set_title(r''+str(j)+' $\delta t$', fontsize=20)
    F.a[1,i].set_xticks([])
    F.a[1,i].set_yticks([])
    F(err,(2,i))
    F.a[2,i].set_title(r''+str(j)+' $\delta t$', fontsize=20)
    F.a[2,i].set_xticks([])
    F.a[2,i].set_yticks([])
    i += 1

#%% 

