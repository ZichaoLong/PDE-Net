"""change 'name' or 'block' to show test errs of different task"""
#%% show errs
import numpy as np
import torch
import conf
from pltutils import *
import matplotlib

name = 'burgers-2-stab0-sparse0.005-msparse0.001-datast1-size5-noise0.001'
block = 18
configfile = 'checkpoint/'+name+'/options.yaml'
options = conf.setoptions(configfile=configfile,isload=True)
print(options['--blocks'])
l = options['--blocks'].index(block)

#%%
errs = torch.load('checkpoint/'+name+'/errs')
a = pltnewaxis()
h = a.figure
err = errs[l].copy()
for i in range(err.shape[1]):
    err[:,i].sort()
b = a.imshow(np.log(err), cmap='jet')
h.colorbar(b, ax=a)
a.set_title('ln(errs)')

#%% 
updateerrs = torch.load('checkpoint/'+name+'/updateerrs')
a = pltnewaxis()
h = a.figure
updateerr = updateerrs[l].copy()
for i in range(updateerr.shape[1]):
    updateerr[:,i].sort()
b = a.imshow(np.log(updateerr), cmap='jet')
h.colorbar(b, ax=a)
a.set_title('ln(updateerrs)')

#%%
uvars = torch.load('checkpoint/'+name+'/uvars')
a = pltnewaxis()
h = a.figure
for i in range(uvars.shape[1]):
    uvars[:,i].sort()
b = a.imshow(np.log(uvars), cmap='jet')
h.colorbar(b, ax=a)
a.set_title('ln(uvars)')

#%%
updatevars = torch.load('checkpoint/'+name+'/updatevars')
a = pltnewaxis()
h = a.figure
for i in range(updatevars.shape[1]):
    updatevars[:,i].sort()
b = a.imshow(updatevars, cmap='jet')
h.colorbar(b, ax=a)
a.set_title('updatevars')

#%%
num = 100
data = np.log(err)
img = np.zeros((num,data.shape[1]))
bins = np.linspace(-15, 1, endpoint=True, num=num+1)
for i in range(img.shape[1]):
    img[:,i] = np.histogram(data[:,i], bins=bins)[0]
# img = np.cumsum(img, axis=0)
img = img[::-1]/err.shape[0]
a = pltnewaxis()
h = a.figure
b = a.imshow(img, cmap='jet')
h.colorbar(b, ax=a)
a.set_title('ln(errs)')
# a.set_yticklabels(['']+list("{:.1f}".format(i) for i in bins[::-25])+[''])
# a.set_yticklabels(['']+list("{:.1f}".format(i) for i in bins[::-50]))
yticklabels = []
yticks = [0, 25, 50, 75, 100]
for i in yticks:
    yticklabels.append(matplotlib.text.Text(0, i, "{:.1f}".format(bins[::-1][i])))
a.set_yticks(yticks)
a.set_yticklabels(yticklabels)
#%%


