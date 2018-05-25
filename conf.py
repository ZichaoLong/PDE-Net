import os,sys,getopt,yaml,contextlib
import numpy as np

def default_options():
    options = {
            '--name':'test',
            '--dtype':'double',
            '--device':'cpu',
            '--constraint':'frozen',
            # computing region
            '--eps':2*np.pi,
            '--dt':1e-2,
            '--cell_num':1,
            '--blocks':'0-6,9,12,15,18',
            # super parameters of network
            '--kernel_size':7,
            '--max_order':2,
            '--dx':2*np.pi/32,
            '--hidden_layers':3,
            '--scheme':'upwind',
            # data generator
            '--dataname':'burgers',
            '--viscosity':0.05,
            '--zoom':4,
            '--max_dt':1e-2/16,
            '--batch_size':28,
            '--data_timescheme':'rk2',
            '--channel_names':'u,v',
            '--freq':4,
            '--data_start_time':1.0,
            # data transform
            '--start_noise':0.001,
            '--end_noise':0.001,
            # others
            '--stablize':0.0,
            '--sparsity':0.001,
            '--momentsparsity':0.001,
            '--npseed':-1,
            '--torchseed':-1,
            '--maxiter':2000,
            '--recordfile':'converge',
            '--recordcycle':200,
            '--savecycle':-1,
            '--start_from':-1,
            }
    return options
def _set_blocks(option, thistype):
    """
    cast string to list of int: 
        0-3,5,10-12 -> [0,1,2,3,5,10,11,12]
        1,2,6 -> [1,2,6]
    """
    if not isinstance(option, str):
        return option
    l0 = option.split(',')
    l = []
    for l1 in l0:
        try:
            ll = thistype(l1)
            x = [ll,]
        except ValueError:
            z = l1.split('-')
            x = list(range(int(z[0]), int(z[1])+1))
        finally:
            l = l+x
    return l
def _setopt(options):
    options.pop('-f',1)
    blocks = list(_set_blocks(options['--blocks'], int))
    # options['--blocks'] is expected to be a list of int but not str
    # by default, options['--blocks'] is a str to be cast, see _set_blocks
    default = default_options()
    for k in default: # type cast
        if isinstance(default[k], (float,int)):
            if isinstance(options[k], str):
                options[k] = eval(options[k])
            if isinstance(default[k], float):
                options[k] = float(options[k])
            else:
                options[k] = int(options[k])
        else:
            options[k] = type(default[k])(options[k])
    options['--blocks'] = blocks
    return None
def setoptions(*, argv=None, kw=None, configfile=None, isload=False):
    """
    proirity: argv>kw>configfile
    Arguments:
        argv (list): command line options
        kw (dict): options
        configfile (str): configfile path
        isload (bool): if True, then options will be loaded from 
            argv/kw/configfile and will not be writen to a configfile,
            otherwise new options will be writen to 
            checkpoint/options['--name']/options.yaml
    """
    options = default_options()
    longopts = list(k[2:]+'=' for k in options)
    longopts.append('configfile=')
    argv = ({} if argv is None else 
            dict(getopt.getopt(argv, shortopts='f',longopts=longopts)[0]))
    kw = ({} if kw is None else kw)
    if '--configfile' in argv:
        assert configfile is None, 'duplicate configfile in argv.'
        configfile = argv['--configfile']

    # update options
    if not configfile is None: # update options from configfile
        options['--configfile'] = configfile
        with open(configfile, 'r') as f:
            options.update(yaml.load(f))
    options.update(kw) # update options from externel parameters
    options.update(argv) # update options from command line 

    # postprocessing options
    _setopt(options)
    savepath = 'checkpoint/'+options['--name']
    if options['--start_from'] >= 0:
        isload = True
    if not isload: # save options to savepath/options.yaml
        try:
            os.makedirs(savepath)
        except FileExistsError:
            os.rename(savepath, savepath+'-'+str(np.random.randint(2**32)))
            os.makedirs(savepath)
        with open(savepath+'/options.yaml', 'w') as f:
            print(yaml.dump(options), file=f)
    return options

