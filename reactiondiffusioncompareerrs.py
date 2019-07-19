"""
compare relative errs: 
change 'name1', 'name2' to compare test errs between different tasks
"""
#%%
from numpy import *
import torch
import matplotlib.pyplot as plt
import conf
name1 = 'reactiondiffusion-frozen-stab0-sparse0.005-msparse0.001-datast0-size5-noise0.001'
name2 = 'reactiondiffusion-2-stab0-sparse0.005-msparse0.001-datast0-size5-noise0.001'
# name3 = 'reactiondiffusion-2-stab1-sparse0.001-msparse0.001-datast0-'+SEEDNAME
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

showblock = [0,2,9,12,15]
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
        y_mean = sqrt(y).mean(axis=1)
        y_up = np.minimum(percentile(sqrt(y),q=upq,axis=0),np.ones(y.shape[1])*1e3)
        y_down = percentile(sqrt(y),q=downq,axis=0)
        ax.flatten()[j].fill_between(x,y_down,y_up,edgecolor=edgecolorlist[s], facecolor=facecolorlist[s],\
                linestyle='-', alpha=alpha)
    if l == 0:
        ax.flatten()[j].set_title(r'warm-up', fontsize=18)
    else:
        ax.flatten()[j].set_title(r''+str(block)+' $\delta t$-block', fontsize=18)
    # ax.flatten()[j].set_yscale('log')
    ax.flatten()[j].set_ylim(1e-3,1.5)
    ax.flatten()[j].set_xticks([1,100,200,300])
    ax.flatten()[j].xaxis.set_tick_params(labelsize=15)
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
        y_mean = sqrt(y).mean(axis=1)
        y_up = np.minimum(percentile(sqrt(y),q=upq,axis=0),np.ones(y.shape[1])*1e3)
        y_down = percentile(sqrt(y),q=downq,axis=0)
        ax.flatten()[j].fill_between(x,y_down,y_up,edgecolor=edgecolorlist[s], facecolor=facecolorlist[s],\
                linestyle='-', alpha=alpha)
    # ax.flatten()[j].set_yscale('log')
    j += 1
#%%

