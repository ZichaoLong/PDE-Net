#%%
import numpy as np
from numpy import *
import torch
from torch.autograd import Variable
import aTEAM
import FD

__all__ = ['VariantCoeLinear2d', 'SingleNonLinear2d']

#%%
class VariantCoeLinear2d(torch.nn.Module):
    def __init__(self, kernel_size, max_order, dx, constraint, xy, interp_degree, interp_mesh_size, dt=1e-2):
        super(VariantCoeLinear2d, self).__init__()
        self.id = FD.FD2d(kernel_size, 0, dx=dx, constraint=constraint, boundary='Periodic')
        self.fd2d = FD.FD2d(kernel_size, max_order, dx=dx, constraint=constraint, boundary='Periodic')
        N = self.fd2d.MomentBank.moment.size()[0]
        self._N = N
        for i in range(N):
            fitter = aTEAM.nn.modules.LagrangeInterpFixInputs(xy, interp_dim=2, interp_degree=interp_degree, mesh_bound=[[0,0],[2*pi,2*pi]], mesh_size=interp_mesh_size)
            fitter.double()
            self.add_module('coe'+str(i), fitter)
        self.register_buffer('dt', torch.DoubleTensor(1).fill_(dt))

    @property
    def coes(self):
        for i in range(self._N):
            yield self.__getattr__('coe'+str(i))
    @property
    def xy(self):
        return Variable(next(self.coes).inputs)
    @xy.setter
    def xy(self, v):
        for fitter in self.coes:
            fitter.inputs = v
    def diff_params(self):
        return list(self.id.parameters())+list(self.fd2d.parameters())
    def coe_params(self):
        params = []
        for coe in self.coes:
            params = params+list(coe.parameters())
        return params
    def forward(self, init, stepnum):
        idkernel = self.id.MomentBank.kernel()
        fdkernel = self.fd2d.MomentBank.kernel()
        coe = []
        for fitter in self.coes:
            coe.append(fitter())
        if init.dim() == 2:
            coe = torch.stack(coe,dim=0)[newaxis,...]
            u = init[newaxis,...]
        else:
            assert init.dim() == 3
            coe = torch.stack(coe, dim=1)
            u = init
        dt = Variable(self.dt)
        for i in range(stepnum):
            uid = self.id(u, idkernel).squeeze(1)
            ufd = self.fd2d(u, fdkernel)
            u = uid+dt*(coe*ufd).sum(dim=1)
        return u.view(init.size())
    def step(self, u):
        return self.forward(u, stepnum=1)

class SingleNonLinear2d(torch.nn.Module):
    def __init__(self, kernel_size, max_order, dx, constraint, xy, interp_degree, interp_mesh_size, nonlinear_interp_degree, nonlinear_interp_mesh_bound, nonlinear_interp_mesh_size, dt=1e-2):
        super(SingleNonLinear2d, self).__init__()
        self.id = FD.FD2d(kernel_size, 0, dx=dx, constraint=constraint, boundary='Dirichlet')
        self.fd2d = FD.FD2d(kernel_size, max_order, dx=dx, constraint=constraint, boundary='Dirichlet')
        N = self.fd2d.MomentBank.moment.size()[0]
        self._N = N
        for i in range(1,N):
            fitter = aTEAM.nn.modules.LagrangeInterpFixInputs(xy, interp_dim=2, interp_degree=interp_degree, mesh_bound=[[0,0],[2*pi,2*pi]], mesh_size=interp_mesh_size)
            fitter.double()
            self.add_module('coe'+str(i), fitter)
        self.nonlinearfitter = aTEAM.nn.modules.LagrangeInterp(interp_dim=1, interp_degree=nonlinear_interp_degree, mesh_bound=nonlinear_interp_mesh_bound, mesh_size=nonlinear_interp_mesh_size)
        self.nonlinearfitter.double()
        self.register_buffer('dt', torch.DoubleTensor(1).fill_(dt))

    @property
    def coes(self):
        for i in range(1, self._N):
            yield self.__getattr__('coe'+str(i))
    @property
    def xy(self):
        return Variable(next(self.coes).inputs)
    @xy.setter
    def xy(self, v):
        for fitter in self.coes:
            fitter.inputs = v
    def diff_params(self):
        return list(self.id.parameters())+list(self.fd2d.parameters())
    def coe_params(self):
        params = []
        for coe in self.coes:
            params = params+list(coe.parameters())
        params = params+list(self.nonlinearfitter.parameters())
        return params
    def forward(self, init, stepnum):
        idkernel = self.id.MomentBank.kernel()
        fdkernel = self.fd2d.MomentBank.kernel()
        coe = []
        for fitter in self.coes:
            coe.append(fitter())
        if init.dim() == 2:
            coe = torch.stack(coe,dim=0)[newaxis,...]
            u = init[newaxis,...]
        else:
            assert init.dim() == 3
            coe = torch.stack(coe,dim=1)
            u = init
        dt = Variable(self.dt)
        for i in range(stepnum):
            uid = self.id(u, idkernel).squeeze(1)
            ufd = self.fd2d(u, fdkernel)
            u = uid+dt*((coe*ufd[:,1:]).sum(dim=1)+self.nonlinearfitter(ufd[:,0]))
        return u.view(init.size())
    def step(self, u):
        return self.forward(u, stepnum=1)
#%%
def test():
    import pdedata
    import torchvision
    trans = torchvision.transforms.Compose([pdedata.DownSample(4), pdedata.ToTensor()])
    d = pdedata.variantcoelinear2d(0.6, mesh_size=[200,200], initfreq=4, transform=trans)
    dataloader = torch.utils.data.DataLoader(d, batch_size=2, num_workers=2)
    dataloader = iter(dataloader)
    sample = next(dataloader)
    sample = pdedata.ToVariable()(sample)
    xy = torch.stack([sample['x'],sample['y']],dim=3)
    u0 = sample['u0']
    linpdelearner = VariantCoeLinear2d(kernel_size=[7,7],max_order=4,dx=2*pi/50,constraint='moment',xy=xy,interp_degree=2,interp_mesh_size=[20,20],dt=1e-2)
    linpdelearner.xy = xy.clone()
    ut = linpdelearner(u0,20)

    trans = torchvision.transforms.Compose([pdedata.DownSample(4, boundary='Dirichlet'), pdedata.ToTensor()])
    d = pdedata.singlenonlinear2d(0.6, mesh_size=[200,200], transform=trans)
    dataloader = torch.utils.data.DataLoader(d, batch_size=2, num_workers=2)
    dataloader = iter(dataloader)
    sample = next(dataloader)
    sample = pdedata.ToVariable()(sample)
    xy = torch.stack([sample['x'],sample['y']],dim=3)
    u0 = sample['u0']
    nonlinearlearner = SingleNonLinear2d(kernel_size=[7,7],max_order=2,dx=2*pi/50,constraint='moment',xy=xy,interp_degree=2,interp_mesh_size=[5,5],nonlinear_interp_degree=3,nonlinear_interp_mesh_bound=[-30,30],nonlinear_interp_mesh_size=40,dt=1e-2)
    nonlinearlearner.xy = xy.clone()
    ut = nonlinearlearner(u0,20)


#%%

