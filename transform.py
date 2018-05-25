import warnings
import numpy as np
from numpy import *
import torch
from scipy.ndimage import gaussian_filter1d

__all__ = ['Compose', 'GaussianSmoothor', 'DownSample', 'AddNoise']

class Compose(object):
    def __init__(self, *args):
        """
        Compose(A,B,C)(sample) = A(B(C(sample)))
        Compose()(sample) = sample
        """
        self.args = []
        for a in args:
            assert callable(a)
            self.args.append(a)
        self.args.reverse()
    def __call__(self, sample):
        for a in self.args:
            sample = a(sample)
        return sample
class GaussianSmoothor(object):
    def __init__(self, sigma, boundary='Periodic', axis=[-1,-2]):
        self.sigma = sigma
        if boundary.upper() == 'PERIODIC':
            self.mode = 'wrap'
        else:
            self.mode = 'constant'
        if isinstance(axis, int):
            self.axis = (axis,)
        else:
            self.axis = tuple(axis)
    def __call__(self, sample):
        if self.sigma < 1e-2:
            return sample
        if sample.requires_grad:
            warnings.warn('lose sample\'s gradient', UserWarning)
        s = sample.cpu().data.numpy()
        for i in self.axis:
            s = gaussian_filter1d(s, self.sigma, axis=i, mode=self.mode)
        sample = torch.from_numpy(s).to(sample)
        return sample
class DownSample(object):
    def __init__(self, mesh_size, axis=[-2,-1], boundary='Periodic'):
        if isinstance(axis, int):
            self.axis = (axis,)
        else:
            self.axis = tuple(axis)
        if isinstance(mesh_size, int):
            mesh_size = [mesh_size,]*len(self.axis)
        self.mesh_size = mesh_size
        self.boundary = boundary
    def __call__(self, sample):
        s = sample
        indx = [slice(None,None),]*sample.dim()
        for i in range(len(self.axis)):
            a = self.axis[i]
            s0 = sample.shape[a]
            s1 = self.mesh_size[i]
            if self.boundary == 'Periodic':
                assert s0%s1 == 0
                slicedownsample = slice(0, None, s0//s1)
            elif self.boundary == 'Dirichlet':
                assert (s0+1)%(s1+1) == 0
                scale = (s0+1)//(s1+1)
                slicedownsample = slice(scale-1,None,scale)
            indx[a] = slicedownsample
            s = s[tuple(indx)]
            indx[a] = slice(None,None)
        return s.clone()
def _keepdim_mean(inputs, axis):
    for i in axis:
        inputs = inputs.mean(dim=i, keepdim=True)
    return inputs
class AddNoise(object):
    def __init__(self, start_noise, end_noise, axis=(-2,-1)):
        self.start_noise = start_noise
        self.end_noise = end_noise
        self.axis = axis
    def __call__(self, start, end=None):
        mean = _keepdim_mean(start, axis=self.axis)
        stdvar = torch.sqrt(_keepdim_mean((start-mean)**2, axis=self.axis))
        startnoise = torch.randn(*start.shape).to(start)
        start = start+self.start_noise*stdvar*startnoise
        if not end is None:
            size = end.shape
            endnoise = torch.randn(*size).to(end)
            end = end+self.end_noise*stdvar*endnoise
            return start, end
        else:
            return start

