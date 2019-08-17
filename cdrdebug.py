"""
debug:
    set options_ref and initialize parameters, test in the following block
compare prediction and err:
    set options_1['--name'] and options_2['--name'] to compare different tasks
"""
#%%
import sys
import time
import numpy as np
import torch
import conf,setenv,initparameters
import matplotlib.pyplot as plt
import aTEAM.pdetools as pdetools
from pltutils import *
#%%
options_ref = configfile_ref = None
options_ref = conf.default_options()
options_ref['--name'] = 'tmp'
options_ref['--dataname'] = 'cdr'
options_ref['--eps'] = 2*np.pi
options_ref['--dt'] = 1e-2
options_ref['--dx'] = 2*np.pi/32
options_ref['--max_dt'] = 1e-4
options_ref['--viscosity'] = 0.1
options_ref['--zoom'] = 4
options_ref['--batch_size'] = 7
options_ref['--data_timescheme'] = 'rk2'
options_ref['--channel_names'] = 'u,v'
options_ref['--freq'] = 4
options_ref = conf.setoptions(argv=None,kw=options_ref,configfile=configfile_ref)
if torch.cuda.is_available():
    options_ref['--device'] = 'cuda'
else:
    options_ref['--device'] = 'cpu'

globalnames_ref, callback_ref, model_ref, data_model_ref, sampling_ref, addnoise_ref = setenv.setenv(options_ref)
globals().update(globalnames_ref)

torch.cuda.manual_seed_all(globalnames_ref['torchseed'])
torch.manual_seed(globalnames_ref['torchseed'])
np.random.seed(globalnames_ref['npseed'])

# initialization of parameters
initparameters.initkernels(model_ref,'upwind')
initparameters.initexpr(model_ref, viscosity=viscosity, pattern=dataname)

# model_ref.polys[k].coeffs(iprint=1)
for poly in model_ref.polys:
    poly.coeffs(iprint=1)

#%%
options_1 = {}
options_1['--name'] = 'cdr-frozen-upwind-sparse0.005-noise0.001'
configfile_1 = 'checkpoint/'+options_1['--name']+'/options.yaml'
options_1 = conf.setoptions(argv=None,kw=None,configfile=configfile_1,isload=True)
if torch.cuda.is_available():
    options_1['--device'] = 'cuda'
else:
    options_1['--device'] = 'cpu'

globalnames_1, callback_1, model_1, data_model_1, sampling_1, addnoise_1 = setenv.setenv(options_1)
globalnames_1['--batch_size'] = 2

callback_1.load(24)

#%%
options_2 = {}
options_2['--name'] = 'cdr-2-upwind-sparse0.005-noise0.001'
configfile_2 = 'checkpoint/'+options_2['--name']+'/options.yaml'
options_2 = conf.setoptions(argv=None,kw=None,configfile=configfile_2,isload=True)
if torch.cuda.is_available():
    options_2['--device'] = 'cuda'
else:
    options_2['--device'] = 'cpu'

globalnames_2, callback_2, model_2, data_model_2, sampling_2, addnoise_2 = setenv.setenv(options_2)
globalnames_2['--batch_size'] = 2

callback_2.load(24)

#%% generate test data
globalnames = globalnames_ref
callback =       callback_ref
model =             model_ref
data_model =   data_model_ref
sampling =       sampling_ref
addnoise =       addnoise_ref
T = 10e-2
batch_size = 2
globalnames['batch_size'] = batch_size
_,_,u = setenv.data(model,data_model,globalnames,sampling,addnoise,block=1,data_start_time=0)
#%% test
init = u[0]
h = plt.figure()
stream0 = h.add_subplot(2,3,1,aspect='equal')
xdiffax = h.add_subplot(2,3,4,aspect='equal')
a0 = h.add_subplot(2,3,2,aspect='equal')
b0 = h.add_subplot(2,3,3,aspect='equal')
a1 = h.add_subplot(2,3,5,aspect='equal')
b1 = h.add_subplot(2,3,6,aspect='equal')
def resetticks(*argv):
    for par in argv:
        par.set_xticks([]); par.set_yticks([])
resetticks(a0,b0,a1,b1)
x1 = init
x0 = sampling(x1)
print(initparameters.trainvar(model.UInputs(x0)))
x0,_ = addnoise(x0,x0)

Y,X = np.mgrid[0:1:(mesh_size[0]+1)*1j,0:1:(mesh_size[1]+1)*1j]
Y,X = Y[:-1,:-1],X[:-1,:-1]
for i in range(40):
    stream0.clear(); xdiffax.clear()
    a0.clear(); a1.clear()
    b0.clear(); b1.clear()

    speed0 = torch.sqrt(x0[0,0]**2+x0[0,1]**2).data.cpu().numpy()
    stream0.streamplot(X,Y,x0[0,0].data.cpu().numpy(),x0[0,1].data.cpu().numpy(),density=0.8,color='k',linewidth=5*speed0/speed0.max())
    timea0 = a0.imshow(x0[0,0].data.cpu().numpy()[::-1], cmap='jet')
    timeb0 = b0.imshow(x0[0,1].data.cpu().numpy()[::-1], cmap='jet')
    timec0 = h.colorbar(timea0, ax=a0)
    timed0 = h.colorbar(timeb0, ax=b0)
    resetticks(a0,b0)
    stream0.set_title('max-min(speed)={:.2f}'.format(speed0.max()-speed0.min()))

    xdiff = torch.sqrt((x0[0,0]-sampling(x1)[0,0])**2+(x0[0,1]-sampling(x1)[0,1])**2)
    xdiffim = xdiffax.imshow(xdiff.data.cpu().numpy()[::-1],cmap='jet')
    specta1 = a1.imshow(sampling(x1)[0,0].data.cpu().numpy()[::-1], cmap='jet')
    spectb1 = b1.imshow(sampling(x1)[0,1].data.cpu().numpy()[::-1], cmap='jet')
    spectc1 = h.colorbar(specta1, ax=a1)
    spectd1 = h.colorbar(spectb1, ax=b1)
    resetticks(a1,b1,xdiffax)
    xdiffax.set_title('max={:.2f},min={:.2f}'.format(xdiff.max().item(),xdiff.min().item()))

    h.suptitle('t={:.1e}'.format(i*T))

    speedrange = max(x1[0,0].max().item()-x1[0,0].min().item(),x1[0,1].max().item()-x1[0,1].min().item())
    relsolutiondiff = (x0-sampling(x1)).abs().max().item()/speedrange

    startt = time.time()
    x0 = model.predict(x0, T=T)
    x0 = x0.data
    with torch.no_grad():
        x1 = data_model.predict(x1, T=T)
    stopt = time.time()
    print('elapsed-time:{:.1f}'.format(stopt-startt)+
            ', speedrange:{:.0f}'.format(speedrange)+
            ', relsolutiondiff:{:.4f}'.format(relsolutiondiff)
            )
    if i > 0:
        timec0.remove()
        timed0.remove()
        spectc1.remove()
        spectd1.remove()
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
v_plot = None
def resetticks(*argv):
    for par in argv:
        par.set_xticks([]); par.set_yticks([])
def showprediction(x_plot, K):
    global u_plot
    global v_plot
    umin=min(x_plot[i][0][0].min() for i in range(len(showstep)))
    umax=max(x_plot[i][0][0].max() for i in range(len(showstep)))
    vmin=min(x_plot[i][0][1].min() for i in range(len(showstep)))
    vmax=max(x_plot[i][0][1].max() for i in range(len(showstep)))
    resetticks(*F0.a.flatten())
    resetticks(*F1.a.flatten())
    for i in range(len(showstep)):
        if sharecolorbar:
            u_plot = F0.a[K,i].imshow(x_plot[i][0],vmin=umin,vmax=umax,cmap='jet')
            v_plot = F1.a[K,i].imshow(x_plot[i][1],vmin=vmin,vmax=vmax,cmap='jet')
        else:
            F0(x_plot[i][0],(K,i))
            F1(x_plot[i][1],(K,i))
        F0.a[K,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=30)
        F1.a[K,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=30)
    if sharecolorbar:
        F0.h.colorbar(u_plot,ax=list(F0.a.flatten()))
        F1.h.colorbar(v_plot,ax=list(F1.a.flatten()))
def showpredictionerrs(x_plot, K):
    global u_plot
    global v_plot
    vmin = -0.1
    vmax = 0.1
    resetticks(*F0.a.flatten())
    resetticks(*F1.a.flatten())
    for i in range(len(showstep)):
        if sharecolorbar:
            u_plot = F0.a[K,i].imshow(x_plot[i][0],vmin=vmin,vmax=vmax,cmap='jet')
            v_plot = F1.a[K,i].imshow(x_plot[i][1],vmin=vmin,vmax=vmax,cmap='jet')
        F0.a[K,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=30)
        F1.a[K,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=30)

#%% show prediction or err step 1
############### set SHOWPREDITIONORERR = 'prediction' or 'err' ###############
SHOWPREDITIONORERR = 'prediction' # 'prediction', 'err'
if SHOWPREDITIONORERR == 'prediction':
    F0 = pltnewmeshbar((3,4))
    F1 = pltnewmeshbar((3,4))
    K = 0
    sharecolorbar = False
    showprediction(x_true, K) # show true solution on K-th row of F0,F1
else:
    sharecolorbar = True
    F0 = pltnewmeshbar((2,4))
    F1 = pltnewmeshbar((2,4))

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
    F1.h.colorbar(v_plot,ax=list(F1.a.flatten()))

# F0 = pltnewmeshbar((3,4))
# F1 = pltnewmeshbar((3,4))
# resetticks(*F0.a.flatten())
# resetticks(*F1.a.flatten())
# for i in range(len(showstep)):
#     err = x_infe[i]-x_true[i]
#     F0(x_true[i][0],(0,i))
#     F0.a[0,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
#     F0(x_infe[i][0], (1,i))
#     F0.a[1,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
#     F0(err[0],(2,i))
#     F0.a[2,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
#     F1(x_true[i][1],(0,i))
#     F1.a[0,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
#     F1(x_infe[i][1], (1,i))
#     F1.a[1,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
#     F1(err[1],(2,i))
#     F1.a[2,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
#%% coeffs
with open('coeffs/coeffs.txt', 'a') as output:
    tsym,csym = model.poly0.coeffs(calprec=8)
    print('name='+globalnames['name'], file=output)
    print('poly0', file=output)
    print(tsym[:8], file=output)
    print(csym[:8], file=output)
    tsym,csym = model.poly1.coeffs(calprec=8)
    print('poly1', file=output)
    print(tsym[:8], file=output)
    print(csym[:8], file=output)


#%%

