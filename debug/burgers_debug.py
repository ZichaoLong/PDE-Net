#%%
import sys
import time
import numpy as np
import torch
import conf,setenv,initparameters
import debug._debug_default_config
#%%
name = 'tmp'
dataname = 'burgers'
eps = 2*np.pi
dt = 1e-2
dx = eps/32
max_dt = 1e-2/16
viscosity = 0.1
zoom = 4
batch_size = 7
data_timescheme = 'rk2'
channel_names = 'u,v'
freq = 4
start_noise = 0.001
end_noise = 0.001
options = configfile = None
options = debug._debug_default_config.default_options()
options['--dataname'] = dataname
options['--eps'] = eps
options['--dt'] = dt
options['--dx'] = dx
options['--max_dt'] = max_dt
options['--viscosity'] = viscosity
options['--zoom'] = zoom
options['--batch_size'] = batch_size
options['--data_timescheme'] = data_timescheme
options['--channel_names'] = channel_names
options['--freq'] = freq
options = conf.setoptions(argv=None,kw=options,configfile=configfile)
if torch.cuda.is_available():
    options['--device'] = 'cuda'
else:
    options['--device'] = 'cpu'

globalnames, callback, model, data_model, sampling, addnoise = setenv.setenv(options)

globals().update(globalnames)

torch.cuda.manual_seed_all(torchseed)
torch.manual_seed(torchseed)
np.random.seed(npseed)

# initialization of parameters
initparameters.initkernels(model,'upwind')
initparameters.initexpr(model, viscosity=viscosity, pattern=dataname)

# model.polys[k].coeffs(iprint=1)
for poly in model.polys:
    poly.coeffs(iprint=1)

#%%
options = {}
options['--name'] = 'burgers-2-size5-noise0.001-test1-sparse0.005-datast1-msparse0.001-stab0-layersparse'
configfile = 'checkpoint/'+options['--name']+'/options.yaml'
options = conf.setoptions(argv=None,kw=None,configfile=configfile,isload=True)
if torch.cuda.is_available():
    options['--device'] = 'cuda'
else:
    options['--device'] = 'cpu'

globalnames, callback, model, data_model, sampling, addnoise = setenv.setenv(options)

globals().update(globalnames)

callback.load(15)

#%% test
T = 10e-2
batch_size = 2
globalnames['batch_size'] = batch_size
import matplotlib.pyplot as plt
import aTEAM.pdetools as pdetools
_,_,u = setenv.data(model,data_model,globalnames,sampling,addnoise,block=1,data_start_time=0)
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
    stream0.streamplot(X,Y,x0[0,0],x0[0,1],density=0.8,color='k',linewidth=5*speed0/speed0.max())
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

#%% show prediction
import matplotlib.pyplot as plt
import aTEAM.pdetools as pdetools
from pltutils import *
_,_,u = setenv.data(model,data_model,globalnames,sampling,addnoise,block=1,data_start_time=0)
x1 = u[0][0]
x0 = sampling(x1)
x0,_ = addnoise(x0,x0)

#%% show prediction step 0
showstep = [0,100,200,300]
x_true = []
for j in showstep:
    with torch.no_grad():
        xtmp = data_model.predict(x1, T=j*dt)
        xtmp = sampling(xtmp)
        _,xtmp = addnoise(x0,xtmp)
        xtmp = xtmp.data.cpu().numpy()
        xtmp = xtmp[:,::-1]
        x_true.append(xtmp)

x_infe = []
for j in showstep:
    with torch.no_grad():
        xtmp = model.predict(x0, T=j*dt)
        xtmp = xtmp.data.cpu().numpy()
        xtmp = xtmp[:,::-1]
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
F1 = pltnewmeshbar((3,4))
#%% show predictions step 3
K = 2
#x_plot = x_true; sharecolorbar = False
x_plot = x_infe; sharecolorbar = False
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
    F0.a[K,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
    F1.a[K,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
if sharecolorbar:
    F0.h.colorbar(u_plot,ax=list(F0.a.flatten()))
    F1.h.colorbar(v_plot,ax=list(F1.a.flatten()))
#%% show predictions step 2 for err
F0 = pltnewmeshbar((2,4))
F1 = pltnewmeshbar((2,4))
#%% show predictions step 3 for err
K = 1
x_plot = err; sharecolorbar = True
vmin = -1
vmax = 1
resetticks(*F0.a.flatten())
resetticks(*F1.a.flatten())
for i in range(len(showstep)):
    if sharecolorbar:
        u_plot = F0.a[K,i].imshow(x_plot[i][0],vmin=vmin,vmax=vmax,cmap='jet')
        v_plot = F1.a[K,i].imshow(x_plot[i][1],vmin=vmin,vmax=vmax,cmap='jet')
    F0.a[K,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
    F1.a[K,i].set_title(r'$T$={:.1f}'.format(showstep[i]*dt), fontsize=20)
#%% show predictions step 4 for err
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
with open('coeffs.txt', 'a') as output:
    tsym,csym = model.poly0.coeffs(calprec=8)
    print(name, file=output)
    print('poly0', file=output)
    print(tsym[:8], file=output)
    print(csym[:8], file=output)
    tsym,csym = model.poly1.coeffs(calprec=8)
    print('poly1', file=output)
    print(tsym[:8], file=output)
    print(csym[:8], file=output)


#%%

