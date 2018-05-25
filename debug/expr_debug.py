#%%
import matlab.engine
eng = matlab.engine.start_matlab()
#%%
import numpy as np
import torch,sympy,time
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
from scipy.optimize.slsqp import fmin_slsqp as slsqp
from scipy.optimize import fmin_bfgs as bfgs
from aTEAM.optim import NumpyFunctionInterface
import expr
from pltutils import *

device = 'cpu'
dtype = torch.float64
polylayers = 3
calprec = 8
scale = 1
noise = 0.001

#%% for burgers
channel_names = 'u,u_x,u_y,u_xx,u_xy,u_yy,v,v_x,v_y,v_xx,v_xy,v_yy'
channel_num = len(channel_names.split(','))
normalization_weight = list(scale**(-(i//2)*2) for i in range(channel_num))
# normalization_weight = ([1,]+[1/(2*np.pi),]*2+[1/(2*np.pi)**2,]*3)*2
rhi = expr.poly(polylayers, channel_num, channel_names=channel_names.split(','), 
        normalization_weight=normalization_weight)
N = 200
viscosity = 0.1
inputs = torch.randn(N, channel_num, dtype=dtype, device=device)
for i in range(channel_num):
    inputs[:,i] /= normalization_weight[i]
outputs_true = -inputs[...,0]*inputs[...,1]-inputs[...,6]*inputs[...,2]+viscosity*inputs[...,3]+viscosity*inputs[...,5]
inputs = inputs+torch.randn(*inputs.shape).to(inputs)*inputs.var(dim=1, keepdim=True)*noise
rhi.layer0.weight.data[0,0] = 1/normalization_weight[0]
rhi.layer0.weight.data[1,1] = 1/normalization_weight[1]
rhi.layer1.weight.data[0,6] = 1/normalization_weight[6]
rhi.layer1.weight.data[1,2] = 1/normalization_weight[2]
rhi.layer_final.weight.data[0,3] = viscosity/normalization_weight[3]
rhi.layer_final.weight.data[0,5] = viscosity/normalization_weight[5]
rhi.layer_final.weight.data[0,12] = -1
rhi.layer_final.weight.data[0,13] = -1
outputs = rhi(inputs)
def forward():
    return ((rhi(inputs)-outputs_true)**2).mean()
nfi = NumpyFunctionInterface(rhi.parameters(), forward=forward, always_refresh=False)
pltnewaxis().plot(nfi.flat_param)
print(((outputs-outputs_true)**2).mean().item())
rhi.coeffs(calprec=calprec)
rhi.coeffs(calprec=calprec,eng=eng)

#%% for heat
channel_names = 'u,u_x,u_y,u_xx,u_xy,u_yy'
channel_num = len(channel_names.split(','))
normalization_weight = list(scale**(-(i//2)*2) for i in range(channel_num))
rhi = expr.poly(polylayers, channel_num, 'u,u_x,u_y,u_xx,u_xy,u_yy'.split(','), normalization_weight=normalization_weight)
N = 100
inputs = torch.randn(N,channel_num, dtype=dtype, device=device)
for i in range(channel_num):
    inputs[:,i] /= normalization_weight[i]
outputs_true = inputs[...,3]+inputs[...,5]
inputs = inputs+torch.randn(*inputs.shape).to(inputs)*inputs.var(dim=1, keepdim=True)*noise
rhi.layer_final.weight.data[0,3] = 1/normalization_weight[3]
rhi.layer_final.weight.data[0,5] = 1/normalization_weight[5]
outputs = rhi(inputs)
def forward():
    return ((rhi(inputs)-outputs_true)**2).mean()
nfi = NumpyFunctionInterface(rhi.parameters(), forward=forward, always_refresh=False)
pltnewaxis().plot(nfi.flat_param)
print(((outputs-outputs_true)**2).mean().item())
rhi.coeffs(calprec=calprec)
rhi.coeffs(calprec=calprec,eng=eng)

#%%
ITERNUM = -1
startt = time.time()
def callback(x):
    global ITERNUM,startt,stopt
    ITERNUM += 1
    if ITERNUM%200 == 0:
        stopt = time.time()
        print('elapsed time: {:.2f}'.format(stopt-startt),'iter:{:6d}'.format(ITERNUM))
        startt = stopt
        print('Func: {:.2e}'.format(nfi.f(x)), ' |g|: {:.2e}'.format(np.linalg.norm(nfi.fprime(x))))
nfi.flat_param = np.random.randn(nfi.numel())/100
xopt = bfgs(nfi.f,nfi.flat_param,nfi.fprime,gtol=1e-16,maxiter=1e6, callback=callback)
xopt,f,d = lbfgsb(nfi.f, nfi.flat_param, nfi.fprime, m=500, factr=1, pgtol=1e-15,maxiter=1e6,iprint=0, callback=callback)
pltnewaxis().plot(nfi.flat_param)
print('lbfgs:{:.2e}'.format(f))
# nfi.flat_param = np.random.randn(nfi.numel())/10
xopt,fx,its,imode,smode = slsqp(nfi.f,nfi.flat_param,fprime=nfi.fprime,acc=1e-16,iter=15000,iprint=1,full_output=True, callback=callback)
print('slsqp:{:.2e}'.format(fx))
startt = time.time()
omat = rhi.expression(calprec=calprec, eng=eng)
stopt = time.time()
print('matlab generate symbolic expression, done! elapsed time: {:.2e}'.format(stopt-startt))
startt = time.time()
osym = rhi.expression(calprec=calprec)
stopt = time.time()
print(' sympy generate symbolic expression, done! elapsed time: {:.3e}'.format(stopt-startt))
# print(osym)
# eng.eval('disp(o)', nargout=0)
startt = time.time()
tmat,cmat = rhi.coeffs(calprec=calprec, eng=eng, o=omat)
stopt = time.time()
print('matlab coeffs coefficients, done! elapsed time: {:.2e}'.format(stopt-startt))
startt = time.time()
tsym,csym = rhi.coeffs(calprec=calprec, o=osym)
stopt = time.time()
print(' sympy coeffs coefficients, done! elapsed time: {:.2e}'.format(stopt-startt))
print(tsym,csym)
print(tmat,cmat)
symboltestinputs = np.random.randn(channel_num)
print('True:  ', rhi(torch.from_numpy(symboltestinputs)).item())
print('mat:   ', rhi.symboleval(symboltestinputs, o=omat, eng=eng))
print('sympy: ', rhi.symboleval(symboltestinputs, o=osym))
#%%


