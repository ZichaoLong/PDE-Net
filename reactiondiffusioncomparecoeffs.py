"""
compare coeffs
change name1, name2 to compare coeffs between different tasks
"""
#%%
from numpy import *
import torch
import matplotlib.pyplot as plt
import conf
name1 = 'reactiondiffusion-2-stab0-sparse0-msparse0.001-datast0-size5-noise0.001-block15'
name2 = 'reactiondiffusion-2-stab0-sparse0.005-msparse0.001-datast0-size5-noise0.001-block15'
D = []
D.append(torch.load('coeffs/'+name1)) # blue
D.append(torch.load('coeffs/'+name2)) # orange
# D.append(torch.load('checkpoint/'+name3+'/errs')) # red
# D.append(torch.load('checkpoint/'+name4+'/errs')) # yellow

coeffs0 = list(d['coeffs0'] for d in D)
coeffs1 = list(d['coeffs1'] for d in D)

edgecolorlist = ('#1B2ACC','#CC4F1B')#, 'red') #, 'yellow')
facecolorlist = ('#089FFF','#FF9848')#, 'red') #, 'yellow')
upq = 90
downq = 10

alpha = 0.25 # facecolor transparency

fig,ax = plt.subplots(1,1)
title = ''
n = 40
x = arange(8,n+1,dtype=float64)
j = 0
i = 0
for s in range(len(edgecolorlist)):
    y = coeffs0[s][:,7:n].copy()
    y[np.isnan(y)] = np.inf
    y_up = percentile(y,q=upq,axis=0)
    y_down = percentile(y,q=downq,axis=0)
    ax.fill_between(x,y_down,y_up,edgecolor=edgecolorlist[s], facecolor=facecolorlist[s],\
            linestyle='-', alpha=alpha)
ax.set_ylim(-0.05,0.05)
ax.grid()
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)

alpha = 0.25 # facecolor transparency

fig,ax = plt.subplots(1,1)
title = ''
n = 40
x = arange(8,n+1,dtype=float64)
j = 0
i = 0
for s in range(len(edgecolorlist)):
    y = coeffs1[s][:,7:n].copy()
    y[np.isnan(y)] = np.inf
    y_up = percentile(y,q=upq,axis=0)
    y_down = percentile(y,q=downq,axis=0)
    ax.fill_between(x,y_down,y_up,edgecolor=edgecolorlist[s], facecolor=facecolorlist[s],\
            linestyle='-', alpha=alpha)
ax.set_ylim(-0.05,0.05)
ax.grid()
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
#%% print coeffs err
import torch
showblock = [2,9,12,15]
Dnosparse = []
Dsparse = []
for k in showblock:
    #Dnosparse.append(torch.load('coeffs/nosparse-'+str(k)))
    Dsparse.append(torch.load('coeffs/reactiondiffusion-2-stab0-sparse0.005-msparse0.001-datast0-size5-noise0.001-block'+str(k)))
    Dnosparse.append(torch.load('coeffs/reactiondiffusion-2-stab0-sparse0-msparse0.001-datast0-size5-noise0.001-block'+str(k)))
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

