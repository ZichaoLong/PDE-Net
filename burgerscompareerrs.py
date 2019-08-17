"""
compare relative errs: 
change 'name1', 'name2' to compare test errs between different tasks
"""
#%%
from numpy import *
import torch
import matplotlib.pyplot as plt
import conf
name1 = 'burgers-frozen-upwind-sparse0.005-noise0.001'
name2 = 'burgers-2-upwind-sparse0.005-noise0.001'
# name3 = 'burgers-2-stab0-sparse0-msparse0.001-datast1-size5-noise0.001'
# name4 = 'coarseburgers'
errs = []
errs.append(torch.load('checkpoint/'+name1+'/errs')) # blue
errs.append(torch.load('checkpoint/'+name2+'/errs')) # orange
# errs.append(torch.load('checkpoint/'+name3+'/errs')) # red
# errs.append(torch.load('checkpoint/'+name4+'/errs')) # yellow
configfile = 'checkpoint/'+name1+'/options.yaml'
options = conf.setoptions(configfile=configfile,isload=True)

edgecolorlist = ('#1B2ACC','#CC4F1B')#, 'red') #, 'yellow')
facecolorlist = ('#089FFF','#FF9848')#, 'red') #, 'yellow')

alpha = 0.25 # facecolor transparency
titlesize = 10
labelsize = 5

showblock = [0,1,2,3,4,5,6,9,12,15,18,21,24,27,30,35,40]
showblockidx = list(options['--blocks'].index(block) for block in showblock)
fig,ax = plt.subplots(1,len(showblock), sharex=False, sharey=True)
title = ''
upq = 100
downq = 25
n = 400
x = arange(1,n,dtype=float64)
j = 0
i = 0
for i in range(len(showblock)):
    l = showblockidx[i]
    block = showblock[i]
    for s in range(len(edgecolorlist)):
        y = errs[s][l][:,1:n].copy()
        y[np.isnan(y)] = np.inf
        y[y>10] = 10
        y_mean = sqrt(y).mean(axis=1)
        y_up = np.minimum(percentile(sqrt(y),q=upq,axis=0),np.ones(y.shape[1])*1e3)
        y_down = percentile(sqrt(y),q=downq,axis=0)
        ax.flatten()[j].fill_between(x,y_down,y_up,edgecolor=edgecolorlist[s], facecolor=facecolorlist[s],\
                linestyle='-', alpha=alpha)
    if l == 0:
        ax.flatten()[j].set_title(r'warm-up', fontsize=titlesize)
    else:
        ax.flatten()[j].set_title(r''+str(block)+' $\delta t$-block', fontsize=titlesize)
    # ax.flatten()[j].set_yscale('log')
    ax.flatten()[j].set_ylim(1e-3,1.5)
    ax.flatten()[j].set_xticks([1,100,200,300])
    ax.flatten()[j].xaxis.set_tick_params(labelsize=labelsize)
    ax.flatten()[j].grid()
    j += 1
ax[0].yaxis.set_tick_params(labelsize=15)
#%%
alpha = 0.5 # facecolor transparency
upq = 75
downq = 25
n = 400
x = arange(1,n,dtype=float64)
j = 0
i = 0
for i in range(len(showblock)):
    l = showblockidx[i]
    block = showblock[i]
    for s in range(len(edgecolorlist)):
        y = errs[s][l][:,1:n].copy()
        y[np.isnan(y)] = np.inf
        y[y>10] = 10
        y_mean = sqrt(y).mean(axis=1)
        y_up = np.minimum(percentile(sqrt(y),q=upq,axis=0),np.ones(y.shape[1])*1e3)
        y_down = percentile(sqrt(y),q=downq,axis=0)
        ax.flatten()[j].fill_between(x,y_down,y_up,edgecolor=edgecolorlist[s], facecolor=facecolorlist[s],\
                linestyle='-', alpha=alpha)
    # ax.flatten()[j].set_yscale('log')
    j += 1
#%%

