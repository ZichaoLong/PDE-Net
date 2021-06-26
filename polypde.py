import numpy as np
from numpy import *
import torch
from torch.autograd import grad
import aTEAM
import aTEAM.utils
from aTEAM.nn.modules import FD
import expr
import initparameters

__all__ = ['POLYPDE2D',]

class POLYPDE2D(torch.nn.Module):
    def __init__(self, dt, dx, kernel_size, max_order, constraint, channel_names, hidden_layers=2, scheme='upwind'):
        super(POLYPDE2D, self).__init__()
        self.register_buffer('dt', torch.DoubleTensor(1).fill_(dt))
        self.dx = dx
        assert max_order >= 1
        self.max_order = max_order
        self._constraint = constraint
        for k in range(max_order+1):
            for j in range(k+1):
                f = FD.FD2d(kernel_size, (j,k-j), dx=self.dx, constraint=constraint, boundary='Periodic')
                f.double()
                self.add_module('fd'+str(j)+str(k-j), f)
        # initparameters.initkernels(self)
        channel_names = channel_names.split(',')
        self.channel_names = channel_names
        self.channel_num = len(channel_names)
        self.hidden_layers = hidden_layers
        allchannels = []
        for c in channel_names:
            for k in range(max_order+1):
                for j in range(k+1):
                    allchannels.append(c+str(j)+str(k-j))
        polys = []
        for k in range(self.channel_num):
            self.add_module('poly'+str(k), expr.poly(hidden_layers, channel_num=len(allchannels), channel_names=allchannels))
            polys.append(self.__getattr__('poly'+str(k)))
        self.polys = tuple(polys)
        self.maxpool = torch.nn.MaxPool2d(self.fd00.moment.shape,stride=1)
        self.relu = torch.nn.ReLU()
        self.scheme = scheme

    @property
    def device(self):
        return self.fd00.moment.device
    @property
    def fds(self):
        for k in range(self.max_order+1):
            for j in range(k+1):
                yield self.__getattr__('fd'+str(j)+str(k-j))
    @property
    def scheme(self):
        return self._scheme
    @scheme.setter
    def scheme(self, v):
        assert v.upper() in ['UPWIND','CENTRAL']
        self._scheme = v
        return v
    def renormalize(self, nw):
        for poly in self.polys:
            poly.renormalize(nw)
        return None
    def expr_params(self):
        parameters = []
        for poly in self.polys:
            parameters += list(poly.parameters())
        return parameters
    def diff_params(self):
        params = []
        for fd in self.fds:
            params += list(fd.parameters())
        return params
    def diff_x_proj(self, *args, **kw):
        for fd in self.fds:
            fd.x_proj()
    def diff_grad_proj(self, *args, **kw):
        for fd in self.fds:
            fd.grad_proj()
    def forward(self, init, T):
        stepnum = round(T/self.dt.item())
        if abs(stepnum-T/self.dt.item())>1e-5:
            raise NotImplementedError('stepnum=T/self.dt.item() should be an integer, but got T={:f} while self.dt={:f}'.format(T,self.dt.item()))
        return self.multistep(init, stepnum)
    def predict(self, init, T):
        return self.forward(init, T)
    def _aggregate_kernel(self):
        k00 = self.fd00.kernel[newaxis,newaxis]*self.fd00.scale
        k01 = self.fd01.kernel[newaxis,newaxis]*self.fd01.scale
        k10 = self.fd10.kernel[newaxis,newaxis]*self.fd10.scale
        k01_flip = -k01.flip([3,])
        k10_flip = -k10.flip([2,])
        kernels = []
        for k in range(2,self.max_order+1):
            for j in range(k+1):
                fd = self.__getattr__('fd'+str(j)+str(k-j))
                kernels.append(fd.kernel[newaxis,newaxis]*fd.scale)
        kernels = torch.cat(kernels, dim=0)
        return k00,k01,k10,k01_flip,k10_flip,kernels
    def RightHandItems(self, u):
        k00,k01,k10,k01_flip,k10_flip,kernels = self._aggregate_kernel()
        assert u.dim() == 4
        conv2d = torch.nn.functional.conv2d
        U = u.split(1,dim=1)
        Upad = list(self.fd00.pad(v) for v in U)
        U00 = list(conv2d(v, k00) for v in Upad)
        U01 = list(conv2d(v, k01) for v in Upad)
        U10 = list(conv2d(v, k10) for v in Upad)
        U01_flip = list(conv2d(v, k01_flip) for v in Upad)
        U10_flip = list(conv2d(v, k10_flip) for v in Upad)
        Uelse = list(conv2d(v, kernels) for v in Upad)
        # uncomment the following code block to enable upwind scheme
        if self.scheme.upper() == 'UPWIND':
            with torch.enable_grad(): # convection direction and update U01[...],U10[...]
                UChannelsTmp = []
                U01Tmp = []; U10Tmp = []
                for k in range(self.channel_num):
                    U01Tmp.append(U01[k].data.clone())
                    U01Tmp[-1].requires_grad = True
                    U10Tmp.append(U10[k].data.clone())
                    U10Tmp[-1].requires_grad = True
                    UChannelsTmp = UChannelsTmp+[U00[k],U01Tmp[k],U10Tmp[k],Uelse[k]]
                UInputsTmp = torch.cat(UChannelsTmp, dim=1)
                UaddTmp = list(poly(UInputsTmp.permute(0,2,3,1))[:,newaxis] for poly in self.polys)
                for k in range(self.channel_num):
                    G = grad([UaddTmp[k].sum(),], [U01Tmp[k],U10Tmp[k]])
                    uk01 = (G[0]>0).to(dtype=u.dtype)*U01[k]+(G[0]<=0).to(dtype=u.dtype)*U01_flip[k]
                    U01[k] = uk01
                    uk10 = (G[1]>0).to(dtype=u.dtype)*U10[k]+(G[1]<=0).to(dtype=u.dtype)*U10_flip[k]
                    U10[k] = uk10
        # comment the above code block to use central scheme
        UChannels = []
        for k in range(self.channel_num):
            UChannels = UChannels+[U00[k],U01[k],U10[k],Uelse[k]]
        UInputs = torch.cat(UChannels, dim=1)
        Uadd = list(poly(UInputs.permute(0,2,3,1))[:,newaxis] for poly in self.polys)
        uadd = torch.cat(Uadd, dim=1)
        return uadd
    def multistep(self, init, stepnum):
        if init.dim() == 2:
            assert self.channel_num == 1
            u = init[newaxis,newaxis,...]
        elif init.dim() == 3:
            if self.channel_num == 1:
                u = init[:,newaxis,]
            else:
                u = init[newaxis]
        else:
            u = init
        assert u.shape[1] == self.channel_num
        #for i in range(stepnum):
        #    uadd = self.RightHandItems(u)
        #    u = u+self.dt*uadd
        #return u.view(init.size())
        for i in range(stepnum):
            uaddhalf = self.RightHandItems(u)
            uhalf = u+(self.dt/2)*uaddhalf
            uadd = self.RightHandItems(uhalf)
            u = u+self.dt*uadd
        return u.view(init.size())
    def step(self, u):
        return self.multistep(u, stepnum=1)
    def UInputs(self, u):
        k00,k01,k10,k01_flip,k10_flip,kernels = self._aggregate_kernel()
        assert u.dim() == 4
        conv2d = torch.nn.functional.conv2d
        U = u.split(1,dim=1)
        Upad = list(self.fd00.pad(v) for v in U)
        U00 = list(conv2d(v, k00) for v in Upad)
        U01 = list(conv2d(v, k01) for v in Upad)
        U10 = list(conv2d(v, k10) for v in Upad)
        Uelse = list(conv2d(v, kernels) for v in Upad)
        UChannels = []
        for k in range(self.channel_num):
            UChannels = UChannels+[U00[k],U01[k],U10[k],Uelse[k]]
        UInputs = torch.cat(UChannels, dim=1)
        return UInputs
    def maximalprinciple(self, u):
        upad = self.fd00.pad(u)
        ut = self.step(u)
        return self.relu(ut-self.maxpool(upad))+self.relu(-ut-self.maxpool(-upad))
