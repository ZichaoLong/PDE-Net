#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
__all__ = ['pltnewaxis', 'pltnewaxis3d', 'pltnewmeshbar']
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from numpy import meshgrid
def pltnewaxis(n=1, m=1):
    f = plt.figure()
    k = 0
    a = ndarray(shape=[n,m], dtype=np.object)
    for i in range(n):
        for j in range(m):
            k += 1
            a[i,j] = f.add_subplot(n,m,k)
    if n*m==1:
        return a[0,0]
    else:
        return a
def pltnewaxis3d(n=1, m=1):
    f = plt.figure()
    k = 0
    a = ndarray(shape=[n,m], dtype=np.object)
    for i in range(n):
        for j in range(m):
            k += 1
            a[i,j] = f.add_subplot(n,m,k, projection='3d')
    if n*m==1:
        return a[0,0]
    else:
        return a
#def pltnewmeshbar(x,y,z,N=50):
#    a = pltnewaxis()
#    X,Y = meshgrid(x,y)
#    b = a.contourf(X, Y, z, N, cmap='jet')
#    a.get_figure().colorbar(b, ax=a)
#    return a
def pltnewmeshbar(shape=(1,1), projection=None):
    import numpy
    h = plt.figure()
    assert isinstance(shape[0], int)
    assert isinstance(shape[1], int)
    a = numpy.ndarray([shape[0],shape[1]], dtype=numpy.object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if projection is None:
                a[i,j] = h.add_subplot(shape[0], shape[1], j+1+i*shape[1])
            else:
                a[i,j] = h.add_subplot(shape[0], shape[1], j+1+i*shape[1], projection='3d')
    def F(im, position=(0,0)):
        if isinstance(position, int):
            ax = a.flat[position]
        else:
            ax = a[position[0], position[1]]
        b = ax.imshow(im, cmap='jet')
        ax.get_figure().colorbar(b, ax=ax)
    F.h = h
    F.a = a
    return F
#%%


