#%%
import sys
import time
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import aTEAM.pdetools as pdetools
import conf,setenv,initparameters
import debug._debug_default_config
dataname = 'heat'
dt = 1e-2
max_dt = 1e-2
zoom = 1
batch_size = 2
data_timescheme = 'euler'
channel_names = 'u'
#%%
options = configfile = None
options = debug._debug_default_config.default_options()
options['--dataname'] = dataname
options['--dt'] = dt
options['--max_dt'] = max_dt
options['--zoom'] = zoom
options['--batch_size'] = batch_size
options['--data_timescheme'] = data_timescheme
options['--channel_names'] = channel_names
options = conf.setoptions(argv=None,kw=options,configfile=configfile)
if torch.cuda.is_available():
    options['--device'] = 'cuda'
else:
    options['--device'] = 'cpu'

globalnames, callback, model, data_model, sampling, addnoise = setenv.setenv(options)

globals().update(globalnames)

# initialization of parameters
initparameters.initkernels(model)
initparameters.initexpr(model, viscosity=viscosity, pattern=dataname)

#%% model.polys[k].coeffs(iprint=1)
for poly in model.polys:
    poly.coeffs(iprint=1)

#%% test
T = 5e-2
init = pdetools.init.initgen(mesh_size=data_model.mesh_size, freq=1, 
        device=device, batch_size=model.channel_num*batch_size)*0.5
init += init.abs().max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]*\
        torch.randn(model.channel_num*batch_size,1,1, dtype=dtype, device=device)*\
        torch.rand(model.channel_num*batch_size,1,1, dtype=dtype, device=device)*2
h = plt.figure()
a0 = h.add_subplot(121)
a1 = h.add_subplot(122)
x0 = pdetools.init.initgen(mesh_size=data_model.mesh_size, freq=freq, device=device, batch_size=batch_size)
x1 = x0
x0 = sampling(x1)
x0,_ = addnoise(x0,x0)
for i in range(1,21):
    startt = time.time()
    x0 = model.predict(x0, T=T)
    x0 = x0.data
    with torch.no_grad():
        x1 = data_model.predict(x1, T=T)
    stopt = time.time()
    print('elapsed-time={:.1f}, sup(|x0-x1|)={:.2f}'.format(stopt-startt, (x0-sampling(x1)).abs().max().item()))
    a0.clear()
    a1.clear()
    xplot0 = x0 if batch_size == 1 else x0[0]
    xplot1 = x1 if batch_size == 1 else x1[0]
    b0 = a0.imshow(xplot0, cmap='jet')
    b1 = a1.imshow(xplot1, cmap='jet')
    a0.set_title('t={:.1e},max={:.2f},min={:.2f}'.format(i*T,x0.max(),x0.min()))
    a1.set_title('t={:.1e},max={:.2f},min={:.2f}'.format(i*T,x1.max(),x1.min()))
    if i > 1:
        c0.remove()
        c1.remove()
    c0 = h.colorbar(b0, ax=a0)
    c1 = h.colorbar(b1, ax=a1)
    plt.pause(1e-3)
#%%

options = {}
options['--name'] = 'heat-2-sparse0.005'
configfile = 'checkpoint/'+options['--name']+'/options.yaml'
options = conf.setoptions(argv=None,kw=None,configfile=configfile,isload=True)
if torch.cuda.is_available():
    options['--device'] = 'cuda'
else:
    options['--device'] = 'cpu'

globalnames, callback, model, data_model, sampling, addnoise = setenv.setenv(options)

globals().update(globalnames)

callback.load(15)


#%% show prediction
import matplotlib.pyplot as plt
import aTEAM.pdetools as pdetools
from pltutils import *
_,_,u = setenv.data(model,data_model,globalnames,sampling,addnoise,block=1,data_start_time=0)
x1 = u[0][0]
x0 = sampling(x1)
x0,_ = addnoise(x0,x0)

# showstep = [0,100,200,300]
# x_true = []
# for j in showstep:
#     with torch.no_grad():
#         xtmp = data_model.predict(x1, T=j*dt)
#         xtmp = sampling(xtmp)
#         _,xtmp = addnoise(x0,xtmp)
#         xtmp = np.concatenate([xtmp[0],xtmp[1]], axis=1)
#         xtmp = xtmp[::-1]
#         x_true.append(xtmp)
# x_infe = []
# for j in showstep:
#     with torch.no_grad():
#         xtmp = model.predict(x0, T=j*dt).data.cpu().numpy()
#         xtmp = np.concatenate([xtmp[0],xtmp[1]], axis=1)
#         xtmp = xtmp[::-1]
#         x_infe.append(xtmp)
# x_true[0] = x_infe[0]
# 
# F = pltnewmeshbar((3,4))
# def resetticks(*argv):
#     for par in argv:
#         par.set_xticks([]); par.set_yticks([])
# resetticks(*F.a.flatten())
# for i in range(len(showstep)):
#     err = x_infe[i]-x_true[i]
#     F(x_true[i],(0,i))
#     F.a[0,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
#     F(x_infe[i], (1,i))
#     F.a[1,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
#     F(err,(2,i))
#     F.a[2,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
showstep = [0,10,20,30]
x_true = []
for j in showstep:
    with torch.no_grad():
        xtmp = data_model.predict(x1, T=j*dt)
        xtmp = sampling(xtmp)
        _,xtmp = addnoise(x0,xtmp)
        xtmp = xtmp.data.cpu().numpy()
        x_true.append(xtmp)

x_infe = []
for j in showstep:
    with torch.no_grad():
        xtmp = model.predict(x0, T=j*dt)
        xtmp = xtmp.data.cpu().numpy()
        x_infe.append(xtmp)
x_true[0] = x_infe[0]
def resetticks(*argv):
    for par in argv:
        par.set_xticks([]); par.set_yticks([])
F0 = pltnewmeshbar((3,4))
resetticks(*F0.a.flatten())
for i in range(len(showstep)):
    err = x_infe[i]-x_true[i]
    F0(x_true[i][0],(0,i))
    F0.a[0,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
    F0(x_infe[i][0], (1,i))
    F0.a[1,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
    F0(err[0],(2,i))
    F0.a[2,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)

#%%

#%% show prediction
import matplotlib.pyplot as plt
import aTEAM.pdetools as pdetools
from pltutils import *
_,_,u = setenv.data(model,data_model,globalnames,sampling,addnoise,block=1,data_start_time=0)
x1 = u[0][0]
x0 = sampling(x1)
x0,_ = addnoise(x0,x0)

#%% show prediction step 0
showstep = [0,50,100,150]
x_true = []
for j in showstep:
    with torch.no_grad():
        xtmp = data_model.predict(x1, T=j*dt)
        xtmp = sampling(xtmp)
        _,xtmp = addnoise(x0,xtmp)
        xtmp = xtmp.data.cpu().numpy()
        x_true.append(xtmp)

x_infe = []
for j in showstep:
    with torch.no_grad():
        xtmp = model.predict(x0, T=j*dt)
        xtmp = xtmp.data.cpu().numpy()
        x_infe.append(xtmp)
x_true[0] = x_infe[0]
def resetticks(*argv):
    for par in argv:
        par.set_xticks([]); par.set_yticks([])
err = []
for i in range(len(showstep)):
    err.append(x_infe[i]-x_true[i])
#%% show predictions step 2
F0 = pltnewmeshbar((3,4))
#%% show predictions step 3
K = 2
#x_plot = x_true; sharecolorbar = False
x_plot = x_infe; sharecolorbar = False
vmin=min(x_plot[i][0][0].min() for i in range(len(showstep)))
vmax=max(x_plot[i][0][0].max() for i in range(len(showstep)))
resetticks(*F0.a.flatten())
for i in range(len(showstep)):
    if sharecolorbar:
        u_plot = F0.a[K,i].imshow(x_plot[i][0],vmin=umin,vmax=umax,cmap='jet')
    else:
        F0(x_plot[i][0],(K,i))
    F0.a[K,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
if sharecolorbar:
    F0.h.colorbar(u_plot,ax=list(F0.a.flatten()))
#%% show predictions step 2 for err
F0 = pltnewmeshbar((2,4))
#%% show predictions step 3 for err
K = 1
x_plot = err; sharecolorbar = True
vmin = -1
vmax = 1
resetticks(*F0.a.flatten())
for i in range(len(showstep)):
    if sharecolorbar:
        u_plot = F0.a[K,i].imshow(x_plot[i][0],vmin=vmin,vmax=vmax,cmap='jet')
    F0.a[K,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
#%% show predictions step 4 for err
if sharecolorbar:
    F0.h.colorbar(u_plot,ax=list(F0.a.flatten()))

#%% coeffs
with open('coeffs.txt', 'a') as output:
    tsym,csym = model.poly0.coeffs(calprec=8)
    print(name, file=output)
    print('poly0', file=output)
    print(tsym[:8], file=output)
    print(csym[:8], file=output)
#%%
