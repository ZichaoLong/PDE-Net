#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import sys,os
import numpy as np
import scipy
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
import torch
import torchvision
import aTEAM
from aTEAM.optim import NumpyFunctionInterface
import pdedata
import pdelearner
import linpdeconfig
#%%
options = { # defalt options
        '--precision':'double',
        '--taskdescriptor':'linpde5x5moment4order-test',
        '--constraint':'moment',
        '--gpu':0,
        '--kernel_size':5,'--max_order':4,
        '--xn':'50','--yn':'50',
        '--interp_degree':2,'--interp_mesh_size':20,
        '--initfreq':4,'--variant_coe_magnitude':1,
        '--batch_size':24,'--teststepnum':80,
        '--maxiter':20000,
        '--dt':0.015,
        '--start_noise_level':0.015,'--end_noise_level':0.015,
        '--layer':list(range(0,21)),
        '--recordfile':'convergence',
        '--recordcycle':200,'--savecycle':10000,
        '--repeatnum':25,
        }
options = linpdeconfig.setoptions(argv=sys.argv[1:],kw=options,configfile=None)

namestobeupdate, callback, linpdelearner = linpdeconfig.setenv(options)

globals().update(namestobeupdate)

#%% training
trans = torchvision.transforms.Compose([pdedata.DownSample(5), pdedata.ToTensor(), pdedata.ToPrecision(precision), pdedata.AddNoise(start_noise_level, end_noise_level)])
for l in layer:
    if l == 0:
        callback.stage = 'warmup'
        isfrozen = (False if constraint == 'free' else True)
    else:
        callback.stage = 'layer-'+str(l)
        if constraint == 'moment' or constraint == 'free':
            isfrozen = False
        elif constraint == 'frozen':
            isfrozen = True
    step = (l if l>=1 else 1)
    # generate layer-l data
    d = pdedata.variantcoelinear2d(step*dt, mesh_size=mesh_size*5, initfreq=initfreq, variant_coe_magnitude=variant_coe_magnitude, transform=trans)
    dataloader = torch.utils.data.DataLoader(d, batch_size=batch_size,num_workers=1)
    dataloader = iter(dataloader)
    sample = pdedata.ToVariable()(pdedata.ToDevice(gpu)(next(dataloader)))
    del dataloader
    xy = torch.stack([sample['x'],sample['y']], dim=3)
    linpdelearner.xy = xy # set xy for pde-net
    # set NumpyFunctionInterface
    mean = sample['u0'].mean()
    var = ((sample['u0']-mean)**2).mean()
    forward = lambda :torch.mean((linpdelearner(sample['u0'], step)-sample['uT'])**2)/var
    def x_proj(*args,**kw):
        linpdelearner.id.MomentBank.x_proj()
        linpdelearner.fd2d.MomentBank.x_proj()
    def grad_proj(*args,**kw):
        linpdelearner.id.MomentBank.grad_proj()
        linpdelearner.fd2d.MomentBank.grad_proj()
    nfi = NumpyFunctionInterface(
            [dict(params=linpdelearner.diff_params(), isfrozen=isfrozen, x_proj=x_proj, grad_proj=grad_proj),
                dict(params=linpdelearner.coe_params(), isfrozen=False, x_proj=None, grad_proj=None)],
            forward=forward, always_refresh=False)
    callback.nfi = nfi
    try:
        # optimize
        xopt,f,d = lbfgsb(nfi.f, nfi.flat_param, nfi.fprime, m=500, callback=callback, factr=1e0,pgtol=1e-16,maxiter=maxiter,iprint=50)
    except RuntimeError as Argument:
        with callback.open() as output:
            print(Argument, file=output) # if overflow then just print and continue
    finally:
        # save parameters
        nfi.flat_param = xopt
        callback.save(xopt, 'final') 

#%%

