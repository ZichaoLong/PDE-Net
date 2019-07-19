import numpy as np
import torch
def initkernels(model, scheme=None):
    """
    initialize convolution kernels: scheme='upwind' or 'central'
    """
    kernels = []
    for i in range(4):
        kernels.append(np.zeros((5,5)))
    if scheme == 'upwind':
        kernels[0][2] = np.array([0,0,-3,4,-1])/2
        kernels[1][:,2] = np.array([0,0,-3,4,-1])/2
    elif scheme == 'central':
        kernels[0][2] = np.array([0,-1,0,1,0])/2
        kernels[1][:,2] = np.array([0,-1,0,1,0])/2
    else:
        pass
    kernels[2][2] = np.array([0,1,-2,1,0])
    kernels[3][:,2] = np.array([0,1,-2,1,0])
    shape = tuple(model.fd01.kernel.shape)
    width = ((shape[0]-5)//2,(shape[1]-5)//2)
    for i in range(4):
        kernels[i] = np.pad(kernels[i], ((width[0],width[0]),(width[1],width[1])), 
                mode='constant', constant_values=0)
    if scheme is None:
        kernels[0] = model.fd01.kernel.data.cpu().numpy()
        kernels[1] = model.fd10.kernel.data.cpu().numpy()
    model.fd01.kernel = kernels[0]
    model.fd10.kernel = kernels[1]
    model.fd02.kernel = kernels[2]
    model.fd20.kernel = kernels[3]
    return None

def initexpr(model, viscosity=0.1, pattern='random'):
    """
    initialize SymNet
    pattern='random' for random initialization
    for debug, one can set:
        pattern='burgers' for burgers equation
        pattern='heat' for heat equation
    """
    rhi = model.polys
    if pattern.upper() == 'RANDOM':
        for poly in rhi:
            for p in poly.parameters():
                p.data = torch.randn(*p.shape,dtype=p.dtype,device=p.device)*1e-1
        return None
    for poly in rhi:
        for p in poly.parameters():
            p.data.fill_(0)
    if pattern.upper() == 'BURGERS':
        rhi[0].layer0.weight.data[0,0] = 1
        rhi[0].layer0.weight.data[1,1] = 1
        rhi[0].layer1.weight.data[0,6] = 1
        rhi[0].layer1.weight.data[1,2] = 1
        rhi[0].layer_final.weight.data[0,3] = viscosity
        rhi[0].layer_final.weight.data[0,5] = viscosity
        rhi[0].layer_final.weight.data[0,12] = -1
        rhi[0].layer_final.weight.data[0,13] = -1
        rhi[1].layer0.weight.data[0,0] = 1
        rhi[1].layer0.weight.data[1,7] = 1
        rhi[1].layer1.weight.data[0,6] = 1
        rhi[1].layer1.weight.data[1,8] = 1
        rhi[1].layer_final.weight.data[0,9] = viscosity
        rhi[1].layer_final.weight.data[0,11] = viscosity
        rhi[1].layer_final.weight.data[0,12] = -1
        rhi[1].layer_final.weight.data[0,13] = -1
    elif pattern.upper() == 'HEAT':
        rhi[0].layer_final.weight.data[0,3] = viscosity
        rhi[0].layer_final.weight.data[0,5] = viscosity
    elif pattern.upper() == 'REACTIONDIFFUSION':
        rhi[0].layer0.weight.data[0,0] = 1 # u
        rhi[0].layer0.weight.data[1,0] = 1 # u
        rhi[0].layer1.weight.data[0,6] = 1 # v
        rhi[0].layer1.weight.data[1,6] = 1 # v
        rhi[0].layer2.weight.data[0,12] = 1 # u^2
        rhi[0].layer2.weight.data[0,13] = 1 # v^2
        rhi[0].layer2.weight.data[1,0] = -1 # -u
        rhi[0].layer2.weight.data[1,6] = 1 # v
        rhi[0].layer_final.weight.data[0,0] = 1 # u
        rhi[0].layer_final.weight.data[0,14] = 1 # (u^2+v^2)(v-u)
        rhi[0].layer_final.weight.data[0,3] = viscosity
        rhi[0].layer_final.weight.data[0,5] = viscosity
        rhi[1].layer0.weight.data[0,0] = 1 # u
        rhi[1].layer0.weight.data[1,0] = 1 # u
        rhi[1].layer1.weight.data[0,6] = 1 # v
        rhi[1].layer1.weight.data[1,6] = 1 # v
        rhi[1].layer2.weight.data[0,12] = 1 # u^2
        rhi[1].layer2.weight.data[0,13] = 1 # v^2
        rhi[1].layer2.weight.data[1,0] = 1 # u
        rhi[1].layer2.weight.data[1,6] = 1 # v
        rhi[1].layer_final.weight.data[0,6] = 1 # v
        rhi[1].layer_final.weight.data[0,14] = -1 # -(u^2+v^2)(u+v)
        rhi[1].layer_final.weight.data[0,9] = viscosity
        rhi[1].layer_final.weight.data[0,11] = viscosity
    return None

def trainmean(inputs):
    mean = inputs.mean(dim=-1,keepdim=True).mean(dim=-2,keepdim=True)
    return mean
def trainvar(inputs):
    """
    calculate variance of inputs to study numerical bahaviors during training
    """
    mean = trainmean(inputs)
    var = ((inputs-mean)**2).mean(dim=-1,keepdim=False).mean(dim=-1,keepdim=False).mean(dim=0,keepdim=False)
    return var

def renormalize(model, u):
    uinputs = model.UInputs(u)
    nw = 1/torch.sqrt(trainvar(uinputs))
    model.renormalize(nw)
    return None
