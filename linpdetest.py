#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import sys,os
import numpy as np
import torch
import torchvision
import pdedata
import linpdeconfig

configfile = 'checkpoint/'+sys.argv[1]+'/options.yaml'

options = linpdeconfig.setoptions(configfile=configfile,isload=True)

namestobeupdate, callback, linpdelearner = linpdeconfig.setenv(options)

globals().update(namestobeupdate)

trans = torchvision.transforms.Compose([pdedata.DownSample(5), pdedata.ToTensor(), pdedata.ToPrecision(precision), pdedata.AddNoise(start_noise_level, end_noise_level)])
errs = np.zeros((len(layer),repeatnum*batch_size,teststepnum))
for j in range(repeatnum):
    print('repeatnum: ', j)
    d = pdedata.variantcoelinear2d(np.array(range(1,teststepnum+1))*dt, mesh_size=mesh_size*5, initfreq=initfreq, variant_coe_magnitude=variant_coe_magnitude, transform=trans)
    dataloader = torch.utils.data.DataLoader(d, batch_size=batch_size,num_workers=8)
    dataloader = iter(dataloader)
    sample = pdedata.ToVariable()(next(dataloader))
    del dataloader
    xy = torch.stack([sample['x'],sample['y']], dim=3)
    linpdelearner.xy = xy # set xy for pde-net
    u0 = sample['u0']
    uT = sample['uT']
    for l in layer:
        print('layer: ', l)
        u = (u0.cuda(gpu) if gpu>=0 else u0)
        callback.load(l)
        linpdelearner.id.MomentBank.moment.volatile = True
        for k in range(teststepnum):
            u = linpdelearner(u, 1)
            mean = uT[:,:,:,k].mean(dim=1,keepdim=True).mean(dim=2,keepdim=True)
            var = ((uT[:,:,:,k]-mean)**2).data.mean(dim=1).mean(dim=1).numpy()
            err = ((u.cpu()-uT[:,:,:,k])**2).data.mean(dim=1).mean(dim=1).numpy()
            errs[l,j*batch_size:(j+1)*batch_size,k] = err/var
torch.save(errs, callback.savepath+'/errs.pkl')
print('test results was saved as: '+callback.savepath+'/errs.pkl')

#%%


