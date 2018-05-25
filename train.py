#%%
import sys,time
import numpy as np
import torch
import timeout_decorator
from aTEAM.optim import NumpyFunctionInterface
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
from scipy.optimize.slsqp import fmin_slsqp as slsqp
from scipy.optimize import fmin_bfgs as bfgs
import conf,setenv,initparameters
#%%
kw = None
# kw = {
#         '--name':'test',
#         '--device':'cuda',
#         '--constraint':'2',
#         # computing region
#         '--eps':2*np.pi,
#         '--dt':1e-2,
#         '--blocks':'0-6,9,12,15,18',
#         # super parameters of network
#         '--kernel_size':7,
#         '--dx':2*np.pi/32,
#         '--hidden_layers':3,
#         '--scheme':'upwind',
#         # data generator
#         '--dataname':'burgers',
#         '--viscosity':0.05,
#         '--zoom':4,
#         '--max_dt':1e-2/16,
#         '--batch_size':28,
#         '--data_timescheme':'rk2',
#         '--channel_names':'u,v',
#         '--freq':4,
#         '--data_start_time':1.0,
#         # data transform
#         '--start_noise':0.001,
#         '--end_noise':0.001,
#         # else
#         '--stablize':0.1,
#         '--sparsity':0.001,
#         '--momentsparsity':0.001,
#         '--npseed':1,
#         '--torchseed':1,
#         '--maxiter':2000,
#         }
options = conf.setoptions(argv=sys.argv[1:],kw=kw,configfile=None)

print(options)
globalnames, callback, model, data_model, sampling, addnoise = setenv.setenv(options)

globals().update(globalnames)

torch.cuda.manual_seed_all(torchseed)
torch.manual_seed(torchseed)
np.random.seed(npseed)

# initialization of parameters
if start_from<0:
    initparameters.initkernels(model, scheme=scheme)
    # initparameters.renormalize(model, u0)
    initparameters.initexpr(model, viscosity=viscosity, pattern='random')
else: # load checkpoint of layer-$start_from
    callback.load(start_from, iternum='final')

#%% train
for block in blocks:
    if block<=start_from:
        continue
    print('')
    print('name: ', name)
    r = np.random.randn()+torch.randn(1,dtype=torch.float64,device=device).item()
    with callback.open() as output:
        print('device: ', device, file=output)
        print('generate a random number to check random seed: ', r, file=output)
    # print('block: ', block)
    if block == 0:
        callback.stage = 'warmup'
        isfrozen = (False if constraint == 'free' else True)
    else:
        callback.stage = 'block-'+str(block)
        isfrozen = False
        if constraint == 'frozen':
            isfrozen = True
    stepnum = block if block>=1 else 1
    layerweight = [1,]*stepnum
    # layerweight = list(1/(stepnum+1-i)**2 for i in range(1,stepnum+1))
    # generate data
    u_obs,u_true,u = \
            setenv.data(model,data_model,globalnames,sampling,addnoise,block,data_start_time)
    print(u_obs[0].shape)
    print(u_obs[0].abs().max())
    print(initparameters.trainvar(model.UInputs(u_obs[0])))
    # set NumpyFunctionInterface
    def forward():
        stableloss,dataloss,sparseloss,momentloss = \
                setenv.loss(model, u_obs, globalnames, block, layerweight)
        if block == 0:
            # for stage='warmup', no regularization term used
            stableloss = 0
            sparseloss = 0
            momentloss = 0
        if constraint == 'frozen':
            momentloss = 0
        loss = stablize*stableloss+dataloss+stepnum*sparsity*sparseloss+stepnum*momentsparsity*momentloss
        if torch.isnan(loss):
            loss = (torch.ones(1,requires_grad=True)/torch.zeros(1)).to(loss)
        return loss
    nfi = NumpyFunctionInterface([
        dict(params=model.diff_params(), isfrozen=isfrozen, 
            x_proj=model.diff_x_proj, grad_proj=model.diff_grad_proj), 
        dict(params=model.expr_params(), 
            isfrozen=False) 
        ], forward=forward, always_refresh=False)
    callback.nfi = nfi
    def callbackhook(_callback, *args):
        # global model,block,u0_obs,T,stable_loss,data_loss,sparse_loss
        stableloss,dataloss,sparseloss,momentloss = \
                setenv.loss(model, u_obs, globalnames, block, layerweight)
        stableloss,dataloss,sparseloss,momentloss = \
                stableloss.item(),dataloss.item(),sparseloss.item(),momentloss.item()
        with _callback.open() as output:
            print("stableloss: {:.2e}".format(stableloss), "  dataloss: {:.2e}".format(dataloss), 
                    "  sparseloss: {:.2e}".format(sparseloss), "momentloss: {:.2e}".format(momentloss), 
                    file=output)
        return None
    callbackhookhandle = callback.register_hook(callbackhook)
    if block == 0:
        callback.save(nfi.flat_param, 'start')
    try:
        # optimize
        xopt = bfgs(nfi.f,nfi.flat_param,nfi.fprime,gtol=2e-16,maxiter=maxiter, callback=callback)
        # xopt,f,d = lbfgsb(nfi.f, nfi.flat_param, nfi.fprime, m=maxiter, callback=callback, factr=1e7, pgtol=1e-8,maxiter=maxiter,iprint=0)
        np.set_printoptions(precision=2, linewidth=90)
        for k in range(max_order+1):
            for j in range(k+1):
                print((model.__getattr__('fd'+str(j)+str(k-j)).moment).data.cpu().numpy())
                print((model.__getattr__('fd'+str(j)+str(k-j)).kernel).data.cpu().numpy())
        for p in model.expr_params():
            print(p.data.cpu().numpy())
    except RuntimeError as Argument:
        with callback.open() as output:
            print(Argument, file=output) # if overflow then just print and continue
    finally:
        # save parameters
        nfi.flat_param = xopt
        callback.save(xopt, 'final') 
        with callback.open() as output:
            print('finally, finish this stage', file=output)
        callback.record(xopt, callback.ITERNUM)
        callbackhookhandle.remove()
        @timeout_decorator.timeout(10)
        def printcoeffs():
            with callback.open() as output:
                print('current expression:', file=output)
                for poly in model.polys:
                    tsym,csym = poly.coeffs()
                    print(tsym[:20], file=output)
                    print(csym[:20], file=output)
        try:
            printcoeffs()
        except timeout_decorator.TimeoutError:
            with callback.open() as output:
                print('Time out', file=output)
#%%
u_obs,u_true,u = \
        setenv.data(model,data_model,globalnames,sampling,addnoise,block=1,data_start_time=0)
with callback.open() as output:
    print(u_obs[0].abs().max(), file=output)
with torch.no_grad():
    with callback.open() as output:
        print(model(u_obs[0], T=50*dt).abs().max(), file=output)
        print(model(u_obs[0], T=300*dt).abs().max(), file=output)
#%%

