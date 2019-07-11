import numpy as np
import torch,contextlib,time,os,sys
import torch.utils.hooks as hooks
from collections import OrderedDict

class setcallback(object):
    def __init__(self, options, nfi=None, module=None, stage=None):
        self.name = options['--name']
        self.recordfile = options['--recordfile'] if options['--recordfile'] != 'None' else None
        self.recordcycle = options['--recordcycle'] 
        self.savecycle = options['--savecycle']
        self.savepath = 'checkpoint/'+self.name
        self.startt = time.time()
        self.Fs = []
        self.Gs = []
        self.ITERNUM = 0
        self._hooks = OrderedDict()

    @property
    def stage(self):
        """
        self.stage: a descriptor(str type) of current training stage
        """
        return self._stage
    @stage.setter
    def stage(self, v):
        self._stage = v
        self.ITERNUM = 0
        with self.open() as output:
            # print('\n', file=output)
            print('current stage is: '+v, file=output)
    @contextlib.contextmanager
    def open(self):
        """
        self.open: define the output record file
        """
        isfile = not (self.recordfile is None or self.recordfile == 'None')
        if isfile:
            output = open(self.savepath+'/'+self.recordfile, 'a')
        else:
            output = sys.stdout
        try:
            yield output
        finally:
            if isfile:
                output.close()
    
    # Just like what callback function usually does in any optimizer, 
    # user need to set self.nfi, self.module, self.stage for 
    # the "setcallback" class, to enable this class to 
    #   1. record object function value(i.e. loss)
    #   2. record norm of gradient of object function
    #   3. save or load optimization variables(i.e. parameters of neural network)

    # remember to set self.nfi, self.module, self.stage
    def save(self, xopt, iternum):
        """
        save parameters
        Notice that xopt is a flatten vector
        (see aTEAM.optim.NumpyFunctionInterface.flat_param)
        which consists all the optimization variables of self.module
        """
        self.nfi.flat_param = xopt
        try:
            os.mkdir(self.savepath+'/params')
        except FileExistsError:
            pass
        filename = self.savepath+'/params/'+str(self.stage)+'-xopt-'+str(iternum)
        torch.save(self.module.state_dict(), filename)
        return None
    def load(self, l, iternum=None):
        """
        load storaged parameters from a file.
        the name of the file from which we will load 
        is determined by l and iternum
        """
        if l == 0:
            stage = 'warmup'
        else:
            stage = 'block-'+str(l)
        if iternum is None:
            iternum = 'final'
        else:
            iternum = str(iternum)
        filename = self.savepath+'/params/'+str(stage)+'-xopt-'+iternum
        params = torch.load(filename, map_location=self.module.device)
        self.module.load_state_dict(params)
        return None
    def record(self, xopt, iternum, **args):
        """
        record iteration information to self.open(): 
            iteration number, elapsed time since last record, 
            object function value, norm of gradient of object function
        hooks in self._hooks will be execute one by one.
        """
        self.Fs.append(self.nfi.f(xopt))
        self.Gs.append(np.linalg.norm(self.nfi.fprime(xopt)))
        stopt = time.time()
        with self.open() as output:
            print('iter:{:6d}'.format(iternum), '   time: {:.2f}'.format(stopt-self.startt), file=output)
            print('Func: {:.2e}'.format(self.Fs[-1]), ' |g|: {:.2e}'.format(self.Gs[-1]), file=output)
        for hook in self._hooks.values():
            hook(self, xopt)
        self.startt = stopt
        return None
    def register_hook(self, hook):
        """
        register hook for self.record
        """
        handle = hooks.RemovableHandle(self._hooks)
        self._hooks[handle.id] = hook
        return handle
    def __call__(self, xopt, **args):
        """
        self.record will be triggerd while self.ITERNUM%self.recordcycle == 0
        self.save will be triggerd while self.ITERNUM%self.savecycle == 0:
        """
        if self.recordcycle>0 and self.ITERNUM%self.recordcycle == 0:
            self.record(xopt, iternum=self.ITERNUM, **args)
        if self.savecycle>0 and self.ITERNUM%self.savecycle == 0:
            self.save(xopt, iternum=self.ITERNUM)
        self.ITERNUM += 1
        return None
