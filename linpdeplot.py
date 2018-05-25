#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import numpy as np
import torch
import torchvision
import pdedata
import linpdeconfig
from torch.autograd import Variable
from pltutils import *

configfile = 'checkpoint/'+'linpde7x7moment4order0.015dt0.015noise-double'+'/options.yaml'

options = linpdeconfig.setoptions(configfile=configfile,isload=True)

namestobeupdate, callback, linpdelearner = linpdeconfig.setenv(options)

globals().update(namestobeupdate)

callback.load(7)

#%% show learned coefficient
x = 2*np.pi*np.arange(200)/200
x,y = np.repeat(x[np.newaxis,:], 200, axis=0),np.repeat(x[:,np.newaxis], 200, axis=1)
x,y = torch.from_numpy(x),torch.from_numpy(y)
xy = Variable(torch.stack([x,y],dim=2))
linpdelearner.xy = xy # set xy for pde-net
a = pltnewaxis(3,5)
k = 0
j = 0
for i in range(15):
    coe = eval('linpdelearner.coe'+str(i))
    coe = coe().data.cpu().numpy()
    ax = a.flatten()[i]
    b = ax.imshow(coe, cmap='jet', vmin=-5, vmax=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('C'+str(k-j)+str(j), fontsize=20)
    c = ax.figure.colorbar(b,ax=ax)
    c.set_ticks([-5,-2.5,0,2.5,5])
    j += 1
    if j > k:
        k += 1
        j = 0

#%% show prediction
trans = torchvision.transforms.Compose([pdedata.DownSample(5), pdedata.ToTensor(), pdedata.ToPrecision(precision), pdedata.AddNoise(start_noise_level, end_noise_level)])
d = pdedata.variantcoelinear2d(np.array(range(1,teststepnum+1))*dt, mesh_size=mesh_size*5, initfreq=initfreq, variant_coe_magnitude=variant_coe_magnitude, transform=trans)
sample = pdedata.ToVariable()(d[0])
xy = torch.stack([sample['x'],sample['y']], dim=2)
u0 = sample['u0']
uT = sample['uT']
u0 = (u0.cuda(gpu) if gpu>=0 else u0)
linpdelearner.id.MomentBank.moment.volatile = True
linpdelearner.xy = xy # set xy for pde-net
F = pltnewmeshbar((3,5))
i = 0
for j in (0,20,40,60,80):
    if j == 0:
        u_true = u0.data.cpu().numpy()
        u = u_true
    else:
        u_true = uT[:,:,j-1].data.cpu().numpy()
        u = linpdelearner(u0, j).data.cpu().numpy()
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

