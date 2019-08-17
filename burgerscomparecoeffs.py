"""
compare coeffs
change name1, name2 to compare coeffs between different tasks
"""
#%%
from numpy import *
import torch
import matplotlib.pyplot as plt
import conf
name1 = 'burgers-2-upwind-sparse0-noise0.001-block6'
name2 = 'burgers-2-upwind-sparse0.005-noise0.001-block6'
D = []
D.append(torch.load('coeffs/burgers/'+name1)) # blue
D.append(torch.load('coeffs/burgers/'+name2)) # orange
# D.append(torch.load('checkpoint/'+name3+'/errs')) # red
# D.append(torch.load('checkpoint/'+name4+'/errs')) # yellow

coeffs0 = list(d['coeffs0'] for d in D)
coeffs1 = list(d['coeffs1'] for d in D)

edgecolorlist = ('#1B2ACC','#CC4F1B')#, 'red') #, 'yellow')
facecolorlist = ('#089FFF','#FF9848')#, 'red') #, 'yellow')
upq = 100
downq = 0

alpha = 0.25 # facecolor transparency

fig,ax = plt.subplots(1,1)
title = ''
n = 40
startidx = 4
valuerange = 0.015
x = arange(startidx+1,n+1,dtype=float64)
j = 0
i = 0
for s in range(len(edgecolorlist)):
    y = coeffs0[s][:,startidx:n].copy()
    y[np.isnan(y)] = np.inf
    y_up = percentile(y,q=upq,axis=0)
    y_down = percentile(y,q=downq,axis=0)
    ax.fill_between(x,y_down,y_up,edgecolor=edgecolorlist[s], facecolor=facecolorlist[s],\
            linestyle='-', alpha=alpha)
ax.set_ylim(-valuerange,valuerange)
ax.grid()
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)

alpha = 0.25 # facecolor transparency

fig,ax = plt.subplots(1,1)
title = ''
n = 40
x = arange(startidx+1,n+1,dtype=float64)
j = 0
i = 0
for s in range(len(edgecolorlist)):
    y = coeffs1[s][:,startidx:n].copy()
    y[np.isnan(y)] = np.inf
    y_up = percentile(y,q=upq,axis=0)
    y_down = percentile(y,q=downq,axis=0)
    ax.fill_between(x,y_down,y_up,edgecolor=edgecolorlist[s], facecolor=facecolorlist[s],\
            linestyle='-', alpha=alpha)
ax.set_ylim(-valuerange,valuerange)
ax.grid()
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
#%% imshow
for s in range(len(edgecolorlist)):
    fig,ax = plt.subplots(1,1)
    y = np.abs(coeffs0[s][:,startidx:n].copy())
    y.sort(axis=0)
    z = ax.imshow(y, cmap='jet', vmin=0, vmax=valuerange)
    fig.colorbar(z, ax=ax)
    # ax.set_title(r'remainder for u-component', fontsize=15)
for s in range(len(edgecolorlist)):
    fig,ax = plt.subplots(1,1)
    y = np.abs(coeffs1[s][:,startidx:n].copy())
    y.sort(axis=0)
    z = ax.imshow(y, cmap='jet', vmin=0, vmax=valuerange)
    fig.colorbar(z, ax=ax)
    # ax.set_title(r'remainder for v-component', fontsize=15)
#%% 
