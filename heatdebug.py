"""
debug:
    set options_ref and initialize parameters, test in the following block
compare prediction and err:
    set options_1['--name'] and options_2['--name'] to compare different tasks
"""
#%%
import sys
import time
from numpy import *
import numpy as np
import torch
import conf,setenv,initparameters
import matplotlib.pyplot as plt
import aTEAM.pdetools as pdetools
from pltutils import *
#%%
options_ref = configfile = None
options_ref = conf.default_options()
options_ref['--dataname'] = 'heat'
options_ref['--viscosity'] = 0.1
options_ref['--dt'] = 1e-2
options_ref['--max_dt'] = 1e-2/16
options_ref['--zoom'] = 4
options_ref['--batch_size'] = 2
options_ref['--data_timescheme'] = 'rk2'
options_ref['--channel_names'] = 'u'
options_ref = conf.setoptions(argv=None,kw=options_ref,configfile=configfile)
if torch.cuda.is_available():
    options_ref['--device'] = 'cuda'
else:
    options_ref['--device'] = 'cpu'

globalnames_ref, callback_ref, model_ref, data_model_ref, sampling_ref, addnoise_ref = setenv.setenv(options_ref)

globals().update(globalnames_ref)

# initialization of parameters
initparameters.initkernels(model_ref)
initparameters.initexpr(model_ref, viscosity=viscosity, pattern=dataname)

# model_ref.polys[k].coeffs(iprint=1)
for poly in model_ref.polys:
    poly.coeffs(iprint=1)

#%%
options_1 = {}
options_1['--name'] = 'heat-2-stab0-sparse0.005-msparse0.001-datast0-size5-noise0.001'
configfile_1 = 'checkpoint/'+options_1['--name']+'/options.yaml'
options_1 = conf.setoptions(argv=None,kw=None,configfile=configfile_1,isload=True)
if torch.cuda.is_available():
    options_1['--device'] = 'cuda'
else:
    options_1['--device'] = 'cpu'

globalnames_1, callback_1, model_1, data_model_1, sampling_1, addnoise_1 = setenv.setenv(options_1)
globalnames_1['--batch_size'] = 2

callback_1.load(15)

#%%
options_2 = {}
options_2['--name'] = 'heat-frozen-stab0-sparse0.005-msparse0.001-datast0-size5-noise0.001'
configfile_2 = 'checkpoint/'+options_2['--name']+'/options.yaml'
options_2 = conf.setoptions(argv=None,kw=None,configfile=configfile_2,isload=True)
if torch.cuda.is_available():
    options_2['--device'] = 'cuda'
else:
    options_2['--device'] = 'cpu'

globalnames_2, callback_2, model_2, data_model_2, sampling_2, addnoise_2 = setenv.setenv(options_2)
globalnames_2['--batch_size'] = 2

callback_2.load(15)

#%% test
globalnames = globalnames_ref
callback =       callback_ref
model =             model_ref
data_model =   data_model_ref
sampling =       sampling_ref
addnoise =       addnoise_ref
T = 2e-2
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
    b0 = a0.imshow(xplot0.cpu().numpy(), cmap='jet')
    b1 = a1.imshow(xplot1.cpu().numpy(), cmap='jet')
    a0.set_title('t={:.1e},max={:.2f},min={:.2f}'.format(i*T,x0.max(),x0.min()))
    a1.set_title('t={:.1e},max={:.2f},min={:.2f}'.format(i*T,x1.max(),x1.min()))
    if i > 1:
        c0.remove()
        c1.remove()
    c0 = h.colorbar(b0, ax=a0)
    c1 = h.colorbar(b1, ax=a1)
    plt.pause(1e-3)

##############################################################
#########          show prediction or show err        ########
##############################################################
##### step 0.1 & 0.2: common code block #####
##### step 1: set SHOWPREDITIONORERR = 'prediction' or 'err' #####

#%% show prediction or err step 0.1
globalnames = globalnames_1
callback =       callback_1
model =             model_1
data_model =   data_model_1
sampling =       sampling_1
addnoise =       addnoise_1
_,_,u = setenv.data(model,data_model,globalnames,sampling,addnoise,block=1,data_start_time=0)
x1 = u[0][0]
x0 = sampling(x1)
x0,_ = addnoise(x0,x0)

showstep = [0,50,100,150]
x_true = []
for j in showstep:
    with torch.no_grad():
        xtmp = data_model.predict(x1, T=j*dt)
        xtmp = sampling(xtmp)
        _,xtmp = addnoise(x0,xtmp)
        xtmp = xtmp.data.cpu().numpy()
        xtmp = xtmp[:,::-1]
        x_true.append(xtmp)
#%% show prediction or err step 0.2
u_plot = None
def resetticks(*argv):
    for par in argv:
        par.set_xticks([]); par.set_yticks([])
def showprediction(x_plot, K):
    global u_plot
    umin=min(x_plot[i][0][0].min() for i in range(len(showstep)))
    umax=max(x_plot[i][0][0].max() for i in range(len(showstep)))
    resetticks(*F0.a.flatten())
    for i in range(len(showstep)):
        if sharecolorbar:
            u_plot = F0.a[K,i].imshow(x_plot[i][0],vmin=umin,vmax=umax,cmap='jet')
        else:
            F0(x_plot[i][0],(K,i))
        F0.a[K,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
    if sharecolorbar:
        F0.h.colorbar(u_plot,ax=list(F0.a.flatten()))
def showpredictionerrs(x_plot, K):
    global u_plot
    vmin = -0.005
    vmax = 0.005
    resetticks(*F0.a.flatten())
    for i in range(len(showstep)):
        if sharecolorbar:
            u_plot = F0.a[K,i].imshow(x_plot[i][0],vmin=vmin,vmax=vmax,cmap='jet')
        F0.a[K,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)

#%% show prediction or err step 1
############### set SHOWPREDITIONORERR = 'prediction' or 'err' ###############
SHOWPREDITIONORERR = 'prediction' # 'prediction', 'err'
if SHOWPREDITIONORERR == 'prediction':
    F0 = pltnewmeshbar((3,4))
    K = 0
    sharecolorbar = False
    showprediction(x_true, K) # show true solution on K-th row of F0,F1
else:
    sharecolorbar = True
    F0 = pltnewmeshbar((2,4))

#%% show prediction or err step 2: set K = 1 or 2, globalnames = globalnames_1 or globalnames_2 etc.
K =                       1
globalnames = globalnames_1
callback =       callback_1
model =             model_1
data_model =   data_model_1
sampling =       sampling_1
addnoise =       addnoise_1
x_infe = []
for j in showstep:
    with torch.no_grad():
        xtmp = model.predict(x0, T=j*dt)
        xtmp = xtmp.data.cpu().numpy()
        xtmp = xtmp[:,::-1]
        x_infe.append(xtmp)
x_true[0] = x_infe[0] # x_true[0] should share noise with x_infe[0]
err = []
for i in range(len(showstep)):
    err.append(x_infe[i]-x_true[i])

if SHOWPREDITIONORERR == 'prediction':
    showprediction(x_infe, K) # show inference solution on K-th row of F0,F1
else:
    showpredictionerrs(err, K-1) # show inference err on (K-1)-th row of F0, F1

#%% show err step 3
if sharecolorbar:
    F0.h.colorbar(u_plot,ax=list(F0.a.flatten()))

#%% coeffs
with open('coeffs/coeffs.txt', 'a') as output:
    tsym,csym = model.poly0.coeffs(calprec=8)
    print(name, file=output)
    print('poly0', file=output)
    print(tsym[:8], file=output)
    print(csym[:8], file=output)
#%%
