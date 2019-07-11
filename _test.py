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

globalnames, callback, model, data_model, sampling, addnoise = setenv.setenv(options)

globals().update(globalnames)

outputfile = callback.savepath+'/test-progress'

DATA = torch.load('checkpoint/data-'+dataname+'-'+str(batch_num),map_location=model.device)
u0_obs,uts_obs,uvars,updatevars = DATA['u0_obs'],DATA['uts_obs'],DATA['uvars'],DATA['updatevars']

#%%
max_blocks = 400
batch_size = 50
errs = np.zeros((len(blocks),batch_size,max_blocks))
updateerrs = errs.copy()
with torch.no_grad():
    startt = time.time()
    for bidx in range(len(blocks)):
        b = blocks[bidx]
        callback.load(b)
        ut_infe = u0_obs
        for stepnum in range(max_blocks):
            ut_infe = model(ut_infe, dt)
            ut_obs = uts_obs[stepnum]
            err = ((ut_infe-ut_obs)**2).mean(dim=-1,keepdim=False).mean(dim=-1,keepdim=False)
            var = uvars[stepnum]
            err = (err.sum(dim=1)/var.sum(dim=1)).data.cpu().numpy()
            errs[bidx, :, stepnum] = err
            updateerr = ((ut_infe-ut_obs)**2).mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False)
            updateerr = (updateerr.sum(dim=1)/updatevars[stepnum].sum(dim=1)).data.cpu().numpy()
            updateerrs[bidx, :, stepnum] = updateerr
    with open(outputfile,'a') as output:
        print('test all {} blocks, elapsed time: {:.2f}'.format(len(blocks), time.time()-startt), file=output)
    torch.save(errs, 'checkpoint/'+options['--name']+'/errs'+str(batch_num))
    torch.save(updateerrs, 'checkpoint/'+options['--name']+'/updateerrs'+str(batch_num))
    uvars = torch.stack(uvars, dim=-1).sum(dim=1).data.cpu().numpy()
    updatevars = torch.stack(updatevars, dim=-1).sum(dim=1).data.cpu().numpy()
    torch.save(uvars, 'checkpoint/'+options['--name']+'/uvars'+str(batch_num))
    torch.save(updatevars, 'checkpoint/'+options['--name']+'/updatevars'+str(batch_num))
with open(outputfile,'a') as output:
    print("errs: trainedblocknum x current_batch_size x T/dt: ", errs.shape, file=output)
    print('errs[-1].max(): ', errs[-1].max(), file=output)
#%%
