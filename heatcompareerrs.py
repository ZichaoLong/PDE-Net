"""
compare relative errs
"""
#%%
from numpy import *
import torch
import matplotlib.pyplot as plt
import conf
name1 = 'heat-frozen-upwind-sparse0.005-noise0.001'
name2 = 'heat-2-upwind-sparse0.005-noise0.001'
# name3 = 
errs = []
errs.append(torch.load('checkpoint/'+name1+'/errs')) # blue
errs.append(torch.load('checkpoint/'+name2+'/errs')) # orange
# errs.append(torch.load('checkpoint/'+name3+'/errs')) # red
configfile = 'checkpoint/'+name1+'/options.yaml'
options = conf.setoptions(configfile=configfile,isload=True)

edgecolorlist = ('#1B2ACC','#CC4F1B') #, 'red') #, 'yellow')
facecolorlist = ('#089FFF','#FF9848') #, 'red') #, 'yellow')

alpha = 0.4 # facecolor transparency
titlesize = 20

showblock = [0,2,6,9]
showblockidx = list(options['--blocks'].index(block) for block in showblock)
fig,ax = plt.subplots(1,len(showblock), sharex=False, sharey=True)
title = ''
upq = 100
downq = 25
n = 40
x = arange(1,n,dtype=float64)
j = 0
i = 0
for i in range(len(showblock)):
    l = showblockidx[i]
    block = showblock[i]
    for s in range(len(edgecolorlist)):
        y = errs[s][l][:,1:n].copy()
        y[np.isnan(y)] = np.inf
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
    ax.flatten()[j].set_ylim(1e-4,1e-2)
    ax.flatten()[j].set_xticks([1,10,20,30])
    ax.flatten()[j].grid()
    j += 1

#%%


