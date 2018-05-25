import warnings
import numpy as np
import torch
import aTEAM.pdetools as pdetools
import aTEAM.pdetools.example.burgers2d as burgers2d
import aTEAM.pdetools.example.cde2d as cde2d
import polypde
import conf,transform,setcallback

__all__ = ['setenv',]

def _set_model(options):

    globalnames = {} # variables need to be exported to training&testing script
    for k in options:
        globalnames[k[2:]] = options[k]
    globalnames['dtype'] = torch.float if globalnames['dtype'] == 'float' else torch.float64
    bound = options['--eps']*options['--cell_num']
    s = bound/options['--dx']
    if abs(s-round(s))>1e-6:
        warnings.warn('cell_num*eps/dx should be an integer but got'+str(s))
    if not globalnames['constraint'].upper() in ['FROZEN','MOMENT','FREE']:
        # using moment matrix with globalnames['constraint']-order approximation
        globalnames['constraint'] = int(globalnames['constraint'])
    globalnames['mesh_size'] = [round(s),]*2
    globalnames['mesh_bound'] = [[0,0],[bound,]*2]
    globalnames['kernel_size'] = [options['--kernel_size'],]*2

    model = polypde.POLYPDE2D(
            dt=globalnames['dt'],
            dx=globalnames['dx'],
            kernel_size=globalnames['kernel_size'],
            max_order=globalnames['max_order'],
            constraint=globalnames['constraint'],
            channel_names=globalnames['channel_names'],
            hidden_layers=globalnames['hidden_layers'],
            scheme=globalnames['scheme']
            ) # build pde-net: a PyTorch module/forward network
    if globalnames['dtype'] == torch.float64:
        model.double()
    else:
        model.float()
    model.to(globalnames['device'])

    if globalnames['npseed'] < 0:
        globalnames['npseed'] = np.random.randint(1e8)
    if globalnames['torchseed'] < 0:
        globalnames['torchseed'] = np.random.randint(1e8)

    callback = setcallback.setcallback(options) 
    # some useful interface, see callback.record, callback.save
    callback.module = model

    return globalnames, callback, model

def setenv(options):
    """
    set training & testing environment
    Returns:
        globalnames(dict): variables need to be exported to training & testing script
        callback(function class): callback function for optimizer
        model(torch.nn.Module): PDE-Net, a torch forward neural network
        data_model(torch.nn.Module): a torch module for data generation
        sampling,addnoise(callable function): data down sample and add noise to data
    """
    globalnames, callback, model = _set_model(options)
    if options['--dataname'] == 'None':
        dataoptions = conf.setoptions(configfile='checkpoint/'+
                options['--dataname']+'/options.yaml', 
                isload=True)
        dataoptions['--start_from'] = 80
        assert options['--cell_num']%dataoptions['--cell_num'] == 0
        dataoptions['--device'] = options['--device']
        dataoptions['--dtype'] = options['--dtype']
        _,_,data_model = _set_model(dataoptions)
        data_model.tiling = options['--cell_num']//dataoptions['--cell_num']
    mesh_size = list(m*globalnames['zoom'] for m in globalnames['mesh_size'])
    mesh_bound = globalnames['mesh_bound']
    viscosity = globalnames['viscosity']
    dx = globalnames['cell_num']*globalnames['eps']/mesh_size[0]
    if options['--dataname'].upper() == 'BURGERS':
        max_dt = globalnames['max_dt']
        data_model = burgers2d.BurgersTime2d(max_dt=max_dt,
                mesh_size=mesh_size,
                mesh_bound=mesh_bound,
                viscosity=viscosity,
                timescheme=globalnames['data_timescheme'],
                )
    elif options['--dataname'].upper() == 'HEAT':
        max_dt = globalnames['max_dt']
        data_model = cde2d.Heat(max_dt=max_dt,
                mesh_size=mesh_size,
                mesh_bound=mesh_bound,
                timescheme=globalnames['data_timescheme']
                )
        data_model.coe[0,2] = data_model.coe[2,0] = viscosity
    data_model.to(device=model.device)
    if globalnames['dtype'] == torch.float64:
        data_model.double()
    else:
        data_model.float()
    sampling = transform.Compose(
            transform.DownSample(mesh_size=globalnames['mesh_size']),
            )
    addnoise = transform.AddNoise(start_noise=options['--start_noise'], end_noise=options['--end_noise'])

    return globalnames, callback, model, data_model, sampling, addnoise

def data(model, data_model, globalnames, sampling, addnoise, block, data_start_time=1):
    """
    generate data 
    Returns:
        u_obs(list of torch.tensor): observed data
        u_true(list of torch.tensor): underlying data
        u(list of torch.tensor): underlying high resolution data
    """
    freq, batch_size, device, dtype, dt = \
            globalnames['freq'], globalnames['batch_size'], \
            globalnames['device'], globalnames['dtype'], globalnames['dt']
    u0 = pdetools.init.initgen(mesh_size=data_model.mesh_size, 
            freq=freq, 
            batch_size=model.channel_num*batch_size, 
            device=device, 
            dtype=dtype)*2
    u0 += 4*(torch.rand(model.channel_num*batch_size,1,1,dtype=dtype,device=device)-0.5)
    u0 = u0.view([batch_size, model.channel_num]+data_model.mesh_size.tolist())
    with torch.no_grad():
        if batch_size>1:
            B = batch_size//2
            u0[:B] = data_model(u0[:B], T=data_start_time)
    u = [u0,]
    u0_true = sampling(u0)
    u_true = [u0_true,]
    u0_obs = addnoise(u0_true)
    u_obs = [u0_obs,]
    stepnum = block if block>=1 else 1
    ut = u0
    with torch.no_grad():
        for k in range(stepnum):
            ut = data_model(ut, T=dt)
            u.append(ut)
            ut_true = sampling(ut)
            u_true.append(ut_true)
            _, ut_obs = addnoise(u0_true, ut_true)
            u_obs.append(ut_obs)
    return u_obs,u_true,u

def _sparse_loss(model):
    """
    SymNet regularization
    """
    loss = 0
    s = 1e-3
    for p in model.expr_params():
        p = p.abs()
        loss = loss+((p<s).to(p)*0.5/s*p**2).sum()+((p>=s).to(p)*(p-s/2)).sum()
    return loss
def _moment_loss(model):
    """
    Moment regularization
    """
    loss = 0
    s = 1e-2
    for p in model.diff_params():
        p = p.abs()
        loss = loss+((p<s).to(p)*0.5/s*p**2).sum()+((p>=s).to(p)*(p-s/2)).sum()
    return loss
def loss(model, u_obs, globalnames, block, layerweight=None):
    if layerweight is None:
        layerweight = [0,]*stepnum
        layerweight[-1] = 1
    dt = globalnames['dt']
    stableloss = 0
    dataloss = 0
    sparseloss = _sparse_loss(model)
    momentloss = _moment_loss(model)
    stepnum = block if block>=1 else 1
    ut = u_obs[0]
    for steps in range(1,stepnum+1):
        uttmp = model(ut, T=dt)
        upad = model.fd00.pad(ut)
        stableloss = stableloss+(model.relu(uttmp-model.maxpool(upad))**2+
                model.relu(-uttmp-model.maxpool(-upad))**2).sum()
        dataloss = dataloss+\
                layerweight[steps-1]*torch.mean(((uttmp-u_obs[steps])/dt)**2)
                # layerweight[steps-1]*torch.mean(((uttmp-u_obs[steps])/(steps*dt))**2)
        ut = uttmp
    return stableloss, dataloss, sparseloss, momentloss

