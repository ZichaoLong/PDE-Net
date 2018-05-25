#%%
import numpy as np
from numpy import *
import numpy.fft as fft
import torch
import torch.utils.data
from torch.autograd import Variable

__all__ = ['initgen', 'variantcoelinear2d', 'singlenonlinear2d', 'DownSample', 'ToTensor', 'ToVariable']

#%% initial value generator
def _initgen_periodic(mesh_size, freq=3):
    dim = len(mesh_size)
    x = random.randn(*mesh_size)
    coe = fft.ifftn(x)
    # set frequency of generated initial value
    freqs = random.randint(freq, 2*freq, size=[dim,])
    # freqs = [10,10]
    for i in range(dim):
        perm = arange(dim, dtype=int32)
        perm[i] = 0
        perm[0] = i
        coe = coe.transpose(*perm)
        coe[freqs[i]+1:-freqs[i]] = 0
        coe = coe.transpose(*perm)
    x = fft.fftn(coe)
    assert linalg.norm(x.imag) < 1e-8
    x = x.real
    return x
def initgen(mesh_size, freq=3, boundary='Periodic'):
    if iterable(freq):
        return freq
    x = _initgen_periodic(mesh_size, freq=freq)
    x = x*100
    if boundary.upper() == 'DIRICHLET':
        dim = x.ndim
        for i in range(dim):
            y = arange(mesh_size[i])/mesh_size[i]
            y = y*(1-y)
            s = ones(dim, dtype=int32)
            s[i] = mesh_size[i]
            y = reshape(y, s)
            x = x*y
        x = x[[slice(1,None),]*dim]
        x = x*16
    return x

#%% base class for numpy pde data generator

class PDESolver(object):
    def step(self, init, dt):
        raise NotImplementedError
    def predict(self, init, T):
        if not hasattr(self, 'max_dt'):
            return self.step(init, T)
        else:
            n = int(ceil(T/self.max_dt))
            dt = T/n
            u = init
            for i in range(n):
                u = self.step(u, dt)
            return u

#%% numpy pde data generator

def _coe_modify(A, B, m):
    A[:m,:m] = B[:m,:m]
    A[:m,-m+1:] = B[:m,-m+1:]
    A[-m+1:,:m] = B[-m+1:,:m]
    A[-m+1:,-m+1:] = B[-m+1:,-m+1:]
    return
class _variantcoelinear2d(PDESolver):
    def __init__(self, spectral_size, max_dt=5e-3, variant_coe_magnitude=1):
        assert isinstance(spectral_size, int)
        N = spectral_size
        self.max_dt = max_dt
        assert N%2 == 0
        self._N = spectral_size
        self._coe_mag = variant_coe_magnitude
        freq_shift_coe = zeros((N,))
        freq_shift_coe[:N//2] = arange(N//2)
        freq_shift_coe[:-N//2-1:-1] = -arange(1, 1+N//2)
        self.K0 = reshape(freq_shift_coe, (N,1))
        self.K1 = reshape(freq_shift_coe, (1,N))
        def b10(x):
            y = reshape(x, [-1,2])
            return self._coe_mag*0.5*reshape(cos(y[:,0])+y[:,1]*(2*pi-y[:,1])*sin(y[:,1]), x.shape[:-1])+0.6
        def b01(x):
            y = reshape(x, [-1,2])
            return self._coe_mag*2*reshape(cos(y[:,0])+sin(y[:,1]), x.shape[:-1])+0.8
        self.a = ndarray([5,5], dtype=np.object)
        self.a[0,0] = lambda x:zeros(x.shape[:-1])
        self.a[0,1] = b01
        self.a[1,0] = b10
        self.a[0,2] = lambda x:zeros(x.shape[:-1])+0.3
        self.a[1,1] = lambda x:zeros(x.shape[:-1])
        self.a[2,0] = lambda x:zeros(x.shape[:-1])+0.2
        b00 = lambda x:zeros(x.shape[:-1])
        self.a[list(range(4)),list(range(3,-1,-1))] = b00
        self.a[list(range(5)),list(range(4,-1,-1))] = b00
        self.a_fourier_coe = ndarray([5,5], dtype=np.object)
        self.a_smooth = ndarray([5,5], dtype=np.object)

        xx = arange(0,2*pi,2*pi/N)
        yy = xx.copy()
        yy,xx = meshgrid(xx,yy)
        xx = expand_dims(xx, axis=-1)
        yy = expand_dims(yy, axis=-1)
        xy = concatenate([xx,yy], axis=2)
        m = N//2
        for k in range(3):
            for j in range(k+1):
                tmp_fourier = fft.ifft2(self.a[j,k-j](xy))
                self.a_fourier_coe[j,k-j] = tmp_fourier
                tmp = zeros([m*3,m*3], dtype=np.complex128)
                _coe_modify(tmp, tmp_fourier, m)
                self.a_smooth[j,k-j] = fft.fft2(tmp).real

    @property
    def spectral_size(self):
        return self._N
    def vc_conv(self, order, coe):
        N = self.spectral_size
        m = N//2
        vc_smooth = self.a_smooth[order[0], order[1]]
        tmp = zeros(vc_smooth.shape, dtype=np.complex128)
        _coe_modify(tmp, coe, m)
        C_aug = fft.ifft2(vc_smooth*fft.fft2(tmp))
        C = zeros(coe.shape, dtype=np.complex128)
        _coe_modify(C, C_aug, m)
        return C
    def rhs_fourier(self, L):
        rhsL = zeros(L.shape, dtype=np.complex128)
        rhsL += self.vc_conv([1,0], -1j*self.K0*L)
        rhsL += self.vc_conv([0,1], -1j*self.K1*L)
        rhsL += self.vc_conv([2,0], -self.K0**2*L)
        rhsL += self.vc_conv([1,1], -self.K0*self.K1*L)
        rhsL += self.vc_conv([0,2], -self.K1**2*L)
        return rhsL
    def step(self, init, dt):
        Y = zeros([self._N,self._N], dtype=np.complex128)
        m = self._N//2
        L = fft.ifft2(init)
        _coe_modify(Y, L, m)
        rhsL1 = self.rhs_fourier(Y)
        rhsL2 = self.rhs_fourier(Y+0.5*dt*rhsL1)
        rhsL3 = self.rhs_fourier(Y+0.5*dt*rhsL2)
        rhsL4 = self.rhs_fourier(Y+dt*rhsL3)

        Y = Y+(rhsL1+2*rhsL2+2*rhsL3+rhsL4)*dt/6
        _coe_modify(L, Y, m)
        x_tmp = fft.fft2(L)
        assert linalg.norm(x_tmp.imag) < 1e-10
        x = x_tmp.real
        return x

class _singlenonlinear2d(PDESolver):
    def __init__(self, dx, diffusivity=0.3, nonlinear_coefficient=5):
        self.dx = dx
        self.nonlinear_coefficient = nonlinear_coefficient
        self.diffusivity = 0.3

    @property
    def max_dt(self):
        return self.dx**2/self.diffusivity/4
    def step(self, init, dt):
        u = np.pad(init, pad_width=1, mode='constant')
        u = (u[1:-1,:-2]+u[:-2,1:-1]+u[1:-1,2:]+u[2:,1:-1]-4*u[1:-1,1:-1])
        u *= self.diffusivity*dt/self.dx**2
        u += np.sin(init)*(self.nonlinear_coefficient*dt)
        u += init
        return u

#%% torch pde dataset
class TorchPDEDataSet(torch.utils.data.Dataset):
    def _xy(self):
        x = 2*pi*arange(self.mesh_size[0])/self.mesh_size[0]
        sample = {}
        if self.boundary.upper() == 'PERIODIC':
            sample['x'] = repeat(x[newaxis,:], self.mesh_size[0], axis=0)
            sample['y'] = repeat(x[:,newaxis], self.mesh_size[0], axis=1)
        else:
            x = x[1:]
            sample['x'] = repeat(x[newaxis,:], self.mesh_size[0]-1, axis=0)
            sample['y'] = repeat(x[:,newaxis], self.mesh_size[0]-1, axis=1)
        return sample
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        init = initgen(mesh_size=self.mesh_size, freq=self.initfreq, boundary=self.boundary)
        sample = {}
        sample['u0'] = init
        sample.update(self._xy())
        if isinstance(self.T, float):
            ut = self.pde.predict(init, self.T)
        else:
            assert isinstance(self.T[0], float)
            n = len(self.T)
            ut = np.zeros(list(init.shape)+[n,])
            u = init
            T = [0,]+list(self.T)
            for i in range(n):
                u = self.pde.predict(u, T[i+1]-T[i])
                ut[:,:,i] = u
        sample['uT'] = ut
        if not self.transform is None:
            sample = self.transform(sample)
        return sample
class variantcoelinear2d(TorchPDEDataSet):
    def __init__(self, T, mesh_size, initfreq=5, spectral_size=30, max_dt=5e-3, variant_coe_magnitude=1, transform=None, size=48):
        self.pde = _variantcoelinear2d(spectral_size=spectral_size, max_dt=max_dt, variant_coe_magnitude=variant_coe_magnitude)
        self.T = T
        if isinstance(mesh_size, int):
            self.mesh_size = [mesh_size,]*2
        else:
            self.mesh_size = mesh_size.copy()
        self.initfreq = initfreq
        self.transform = transform
        self.boundary = 'Periodic'
        self.size = size
class singlenonlinear2d(TorchPDEDataSet):
    def __init__(self, T, mesh_size, initfreq=5, diffusivity=0.3, nonlinear_coefficient=5, transform=None, size=48):
        if isinstance(mesh_size, int):
            self.mesh_size = [mesh_size,]*2
        else:
            assert mesh_size[0] == mesh_size[1]
            self.mesh_size = mesh_size[:2]
        dx = 2*pi/mesh_size[0]
        self.T = T
        self.initfreq = initfreq
        self.pde = _singlenonlinear2d(dx=dx,diffusivity=diffusivity, nonlinear_coefficient=nonlinear_coefficient)
        self.transform = transform
        self.boundary = 'Dirichlet'
        self.size = size
#%% torch dataset transform tools
class DownSample(object):
    def __init__(self, scale, boundary='Periodic'):
        assert isinstance(scale, int)
        self.scale = scale
        self.boundary = boundary
    def __call__(self, sample):
        if self.boundary == 'Periodic':
            idx1 = slice(random.randint(self.scale), None, self.scale)
            idx2 = slice(random.randint(self.scale), None, self.scale)
        else:
            idx1 = slice(self.scale-1, None, self.scale)
            idx2 = slice(self.scale-1, None, self.scale)
        s = {}
        for k in sample:
            s[k] = sample[k][idx1,idx2]
        return s
class ToTensor(object):
    def __call__(self, sample):
        s = {}
        for k in sample:
            s[k] = torch.from_numpy(sample[k])
        return s
class ToVariable(object):
    def __call__(self, sample):
        s = {}
        for k in sample:
            s[k] = torch.autograd.Variable(sample[k])
        return s
class ToDevice(object):
    def __init__(self, device):
        assert isinstance(device, int)
        self.device = device
    def __call__(self, sample):
        s = {}
        for k in sample:
            if self.device >= 0:
                s[k] = sample[k].cuda(self.device)
            else:
                s[k] = sample[k].cpu()
        return s
class ToPrecision(object):
    def __init__(self, precision):
        assert precision in ['float','double']
        self.precision = precision
    def __call__(self, sample):
        s = {}
        for k in sample:
            if self.precision == 'float':
                s[k] = sample[k].float()
            else:
                s[k] = sample[k].double()
        return s
class AddNoise(object):
    def __init__(self, start_noise_level, end_noise_level):
        self.start_noise_level = start_noise_level
        self.end_noise_level = end_noise_level
    def __call__(self, sample):
        s = {}
        for k in sample:
            s[k] = sample[k]
        mean = sample['u0'].mean()
        stdvar = sqrt(((sample['u0']-mean)**2).mean())
        size = sample['u0'].size()
        startnoise = sample['u0'].new(size).normal_()
        s['u0'] = sample['u0']+self.start_noise_level*stdvar*startnoise
        if 'uT' in sample:
            size = sample['uT'].size()
            endnoise = sample['uT'].new(size).normal_()
            s['uT'] = sample['uT']+self.end_noise_level*stdvar*endnoise
        return s
#%%
def test_variantcoelinearpde2d():
    import matplotlib.pyplot as plt
    h = plt.figure()
    a = h.add_subplot(111)
    init = initgen(mesh_size=(100,100), freq=5)
    linpde = _variantcoelinear2d(spectral_size=30, max_dt=1e-2, variant_coe_magnitude=1)
    x = init
    for i in arange(0,1,linpde.max_dt):
        x = linpde.step(x, dt=linpde.max_dt)
        a.clear()
        b = a.imshow(x, cmap='jet')
        a.set_title('t={:.2f}'.format(i))
        c = h.colorbar(b, ax=a)
        plt.pause(1e-3)
        c.remove()
    c = h.colorbar(b, ax=a)
def test_singlenonlinear2d():
    import matplotlib.pyplot as plt
    h = plt.figure()
    a = h.add_subplot(111)
    init = initgen(mesh_size=(100,100), freq=3, boundary='Dirichlet')
    sinpde = _singlenonlinear2d(dx=2*pi/50,diffusivity=0.3, nonlinear_coefficient=5)
    x = init
    for i in arange(0,1,sinpde.max_dt):
        x = sinpde.step(x, dt=sinpde.max_dt)
        a.clear()
        b = a.imshow(x, cmap='jet')
        a.set_title('t={:.2f}'.format(i))
        c = h.colorbar(b, ax=a)
        plt.pause(1e-3)
        c.remove()
    c = h.colorbar(b, ax=a)
def test_dataset():
    import torchvision
    trans = torchvision.transforms.Compose([DownSample(4), ToTensor(), AddNoise(0.01,0.01)])
    d = variantcoelinear2d(0.6, mesh_size=[200,200], initfreq=4, transform=trans)
    dataloader = torch.utils.data.DataLoader(d, batch_size=2, num_workers=2)
    dataloader = iter(dataloader)
    sample = next(dataloader)
    trans = torchvision.transforms.Compose([DownSample(4, boundary='Dirichlet'), ToTensor()])
    d = singlenonlinear2d(0.6, mesh_size=[200,200], transform=trans)
    dataloader = torch.utils.data.DataLoader(d, batch_size=2, num_workers=2)
    dataloader = iter(dataloader)
    sample = next(dataloader)
#%%

