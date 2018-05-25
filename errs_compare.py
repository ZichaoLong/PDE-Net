#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare relative errs
"""
#%% 
from numpy import *
import torch
import matplotlib.pyplot as plt
errs = []
errs.append(torch.load('checkpoint/linpde5x5moment4order0.015dt0.015noise-double/errs.pkl')) # blue
errs.append(torch.load('checkpoint/linpde7x7moment4order0.015dt0.015noise-double/errs.pkl')) # orange

edgecolorlist = ('#1B2ACC','#CC4F1B') # , 'red', 'yellow')
facecolorlist = ('#089FFF','#FF9848') # , 'red', 'yellow')

alpha = 0.7 # facecolor transparency

showlayer = [0,7,10,15]
fig,ax = plt.subplots(1,len(showlayer), sharex=False, sharey=True)
title = ''
upq = 75
downq = 25
n = 80
x = arange(1,n,dtype=float64)
j = 0
i = 0
for l in showlayer:
    ll = l
    if l == 0:
        ll = 'warmup'
    else:
        ll = 'layer-'+str(l)
    for s in range(len(edgecolorlist)):
        y = errs[s][l][:,1:n]
        y_mean = sqrt(y).mean(axis=1)
        y_up = percentile(sqrt(y),q=upq,axis=0)
        y_down = percentile(sqrt(y),q=downq,axis=0)
        ax.flatten()[j].fill_between(x,y_down,y_up,edgecolor=edgecolorlist[s], facecolor=facecolorlist[s],\
                linestyle='-', alpha=alpha)
    if l == 0:
        ax.flatten()[j].set_title(r'warm-up', fontsize=20)
    else:
        ax.flatten()[j].set_title(r''+str(l)+' $\delta t$-block', fontsize=20)
    ax.flatten()[j].set_yscale('log')
    ax.flatten()[j].set_ylim(1e-2,1e2)
    ax.flatten()[j].set_xticks([1,20,40,60])
    ax.flatten()[j].grid()
    j += 1

#%%


