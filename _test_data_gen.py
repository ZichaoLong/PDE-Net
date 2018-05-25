import sys,os,time
import numpy as np
import torch
import conf,setenv

configfile = sys.argv[1]
configfile = 'checkpoint/'+configfile+'/options.yaml'
print('configfile: ', configfile)
options = conf.setoptions(configfile=configfile,isload=True)
if not torch.cuda.is_available():
    options['--device'] = 'cpu'
if len(sys.argv)>2:
    batch_num = int(sys.argv[2])
else:
    batch_num = 0
if len(sys.argv)>3:
    options['--device'] = (sys.argv[3] 
            if sys.argv[3].startswith('cuda') or sys.argv[3].startswith('cpu') 
            else 'cuda:'+sys.argv[3])

globalnames, _, model, data_model, sampling, addnoise = setenv.setenv(options)

globals().update(globalnames)

if len(sys.argv)>4:
    npseed = int(sys.argv[4])
else:
    npseed = np.random.get_state()[1][0]
if len(sys.argv)>5:
    torchseed = int(sys.argv[5])
else:
    torchseed = torch.initial_seed()

torch.cuda.manual_seed_all(torchseed)
torch.manual_seed(torchseed)
np.random.seed(npseed)

outputfile = 'checkpoint/data-generator-progress'
r = np.random.randn()+torch.randn(1,dtype=torch.float64,device=device).item()
with open(outputfile, 'a') as output:
    print('device: ', device, file=output)
    print('generate a random number to check random seed: ', r, file=output)

max_blocks = 400
batch_size = 50
globalnames['batch_size'] = batch_size
with torch.no_grad():
    # test data, data_start_time=0
    u_obs,u_true,u = \
            setenv.data(model,data_model,globalnames,sampling,addnoise,1,data_start_time=0)
    u0_obs = u_obs[0]
    ut = u[0]
    mean0 = u0_obs.mean(dim=-2,keepdim=True).mean(dim=-1,keepdim=True)
    uvar0 = ((u0_obs-mean0)**2).mean(dim=-2,keepdim=True).mean(dim=-1,keepdim=True)
    std0 = torch.sqrt(uvar0)
    uts_obs = []
    uvars = []
    updatevars = []
    startt = tmpstartt = time.time()
    for stepnum in range(max_blocks):
        if stepnum%10 == 0 and stepnum != 0:
            with open(outputfile,'a') as output:
                print('n*dt / N*dt: {}/{}, elapsed time: {:.2f}'.format(
                    stepnum,max_blocks,time.time()-tmpstartt), file=output)
            tmpstartt = time.time()
        ut = data_model(ut, dt)
        ut_obs = sampling(ut)
        if start_noise != 0:
            ut_obs = ut_obs+start_noise*std0*torch.randn(*ut_obs.shape, dtype=dtype, device=device)
        uts_obs.append(ut_obs)
        meant = ut_obs.mean(dim=-2,keepdim=True).mean(dim=-1,keepdim=True)
        uvart = ((ut_obs-meant)**2).mean(dim=-1,keepdim=False).mean(dim=-1,keepdim=False)
        uvars.append(uvart)
        updatemeant = (ut_obs-u0_obs).mean(dim=-2,keepdim=True).mean(dim=-1,keepdim=True)
        updatevart = ((ut_obs-u0_obs-updatemeant)**2).mean(dim=-1,keepdim=False).mean(dim=-1,keepdim=False)
        updatevars.append(updatevart)
    with open(outputfile,'a') as output:
        print('GENERATE DATA, elapsed time: {:.2f}'.format(time.time()-startt), file=output)
torch.save(dict(u0_obs=u0_obs,uts_obs=uts_obs,uvars=uvars,updatevars=updatevars), 
        'checkpoint/data-'+dataname+'-'+str(batch_num)) # +'-'+str(npseed)+'-'+str(torchseed))
