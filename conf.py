import os,sys,getopt,yaml,contextlib
import numpy as np

def default_options():
    options = {
            '--name':'test',                # case name, string
            '--dtype':'double',             # string: 'double', 'float'
            '--device':'cpu',               # select device for the model, a string: 'cpu', 'cuda', 'cuda:0', 'cuda:3' etc.
            '--constraint':'2',        # constraint type of convolution kernel: 'frozen'(string), 1 (int, 1st order precision), 2 (int, 2nd order precision), etc.
            # computing region              # 
            '--dt':1e-2,                    # time step size of the learned model, double
            '--cell_num':1,                 # compute region: [0,eps*cell_num]**2, int
            '--eps':2*np.pi,                # compute region: [0,eps*cell_num]**2, double
            '--blocks':'0-6,9,12,15',       # training blocks: 0 for warmup, a string to be convert to a list of int
            # super parameters of network   # 
            '--kernel_size':5,              # convolution kernel size, int
            '--max_order':2,                # max spatial differential order in the model, int
            '--dx':2*np.pi/32,              # delta x, double
            '--hidden_layers':5,            # hidden layers of symnet, int
            '--scheme':'upwind',            # string: upwind, central
            # data generator                # 
            '--dataname':'burgers',         # dataname, string: burgers, heat, cdr
            '--viscosity':0.05,             # double
            '--zoom':4,                     # dx(of data generator) = dx(of learned model)/zoom, double
            '--max_dt':1e-2/16,             # max dt of data generator, double
            '--batch_size':28,              # batch size, int
            '--data_timescheme':'rk2',      # time scheme for data generator, string: rk2, euler
            '--channel_names':'u,v',        # 
            '--freq':4,                     # initial data frequency, int
            '--data_start_time':1.0,        # 
            # data transform                # 
            '--start_noise':0.001,          # noise of initial value
            '--end_noise':0.001,            # noise of end time value
            # others                        # 
            '--stablize':0.0,               # 
            '--sparsity':0.005,             # sparsity regularization on parameters of symnet
            '--momentsparsity':0.001,       # moment sparsity regularization on parameters of moment matrix
            '--npseed':-1,                  # numpy random seed, -1 means no specific random seed
            '--torchseed':-1,               # torch random seed, -1 means no specifig random seed
            '--maxiter':2000,               # maxiteration of each stage of training
            '--recordfile':'converge',      # converge information of each stage will be print into checkpoint/${name}/converge
            '--recordcycle':200,            # print information each 'recordcycle' steps during training
            '--savecycle':-1,               # 
            '--start_from':-1,              # 
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
            options.update(yaml.load(f,Loader=yaml.FullLoader))
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

