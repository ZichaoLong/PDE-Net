"""
compare coeffs
"""
#%%
from numpy import *
import torch
import matplotlib.pyplot as plt
import conf
D = []
D.append(torch.load('coeffs/nosparse-15')) # blue
D.append(torch.load('coeffs/sparse-15')) # orange
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
x = arange(5,n+1,dtype=float64)
j = 0
i = 0
for s in range(len(edgecolorlist)):
    y = coeffs0[s][:,4:n].copy()
    y[np.isnan(y)] = np.inf
    y_up = percentile(y,q=upq,axis=0)
    y_down = percentile(y,q=downq,axis=0)
    ax.fill_between(x,y_down,y_up,edgecolor=edgecolorlist[s], facecolor=facecolorlist[s],\
            linestyle='-', alpha=alpha)
ax.set_ylim(-0.015,0.015)
ax.grid()
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)

alpha = 0.25 # facecolor transparency

fig,ax = plt.subplots(1,1)
title = ''
n = 40
x = arange(5,n+1,dtype=float64)
j = 0
i = 0
for s in range(len(edgecolorlist)):
    y = coeffs1[s][:,4:n].copy()
    y[np.isnan(y)] = np.inf
    y_up = percentile(y,q=upq,axis=0)
    y_down = percentile(y,q=downq,axis=0)
    ax.fill_between(x,y_down,y_up,edgecolor=edgecolorlist[s], facecolor=facecolorlist[s],\
            linestyle='-', alpha=alpha)
ax.set_ylim(-0.015,0.015)
ax.grid()
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
#%% print coeffs err
import torch
showblock = [1,2,9,12,15]
Dnosparse = []
Dsparse = []
for k in showblock:
    #Dnosparse.append(torch.load('coeffs/nosparse-'+str(k)))
    Dsparse.append(torch.load('coeffs/sparse-'+str(k)))
#%% 
def normalize(c):
    c = c.copy()
    c[:,:2] = c[:,:2]+1
    c[:,2:4] = (c[:,2:4]-0.05)/0.05
    return c
with open('coeffs/coeffserr.txt','a') as output:
    print(*showblock,file=output)
# nosparse err
#D = Dnosparse
#with open('coeffs/coeffserr.txt','a') as output:
#    print('nosparse',file=output)
D = Dsparse
with open('coeffs/coeffserr.txt','a') as output:
    print('sparse',file=output)

with open('coeffs/coeffserr.txt','a') as output:
    print('model', file=output)
err = []
for i in range(len(showblock)):
    coeffs0 = normalize(D[i]['coeffs0'])
    coeffs1 = normalize(D[i]['coeffs1'])
    coeffs = np.concatenate((coeffs0,coeffs1))
    err.append(np.sqrt((coeffs[:,:4]**2).mean()))
with open('coeffs/coeffserr.txt','a') as output:
    print(err,file=output)
with open('coeffs/coeffserr.txt','a') as output:
    print('remainder', file=output)
err = []
for i in range(len(showblock)):
    coeffs0 = normalize(D[i]['coeffs0'])
    coeffs1 = normalize(D[i]['coeffs1'])
    coeffs = np.concatenate((coeffs0,coeffs1))
    err.append(np.sqrt((coeffs[:,4:]**2).mean()))
with open('coeffs/coeffserr.txt','a') as output:
    print(err,file=output)


#%%

